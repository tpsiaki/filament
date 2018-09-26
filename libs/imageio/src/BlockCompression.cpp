/*
 * Copyright (C) 2018 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <imageio/BlockCompression.h>

#include <cmath>

using namespace image;

using std::string;

// ---------------------------------------------------------------------------------------------
// Declare constants, enums, and structs that are used when interfacing with astcenc
// ---------------------------------------------------------------------------------------------

#define MAX_TEXELS_PER_BLOCK 216
#define PARTITION_BITS 10
#define PARTITION_COUNT (1 << PARTITION_BITS)

enum astc_decode_mode {
    DECODE_LDR_SRGB,
    DECODE_LDR,
    DECODE_HDR
};

struct astc_codec_image {
    uint8_t*** imagedata8;
    uint16_t*** imagedata16;
    int xsize;
    int ysize;
    int zsize;
    int padding;
};

struct error_weighting_params {
    float rgb_power;
    float rgb_base_weight;
    float rgb_mean_weight;
    float rgb_stdev_weight;
    float alpha_power;
    float alpha_base_weight;
    float alpha_mean_weight;
    float alpha_stdev_weight;
    float rgb_mean_and_stdev_mixing;
    int mean_stdev_radius;
    int enable_rgb_scale_with_alpha;
    int alpha_radius;
    int ra_normal_angular_scale;
    float block_artifact_suppression;
    float rgba_weights[4];
    float block_artifact_suppression_expanded[MAX_TEXELS_PER_BLOCK];
    int partition_search_limit;
    float block_mode_cutoff;
    float texel_avg_error_limit;
    float partition_1_to_2_limit;
    float lowest_correlation_cutoff;
    int max_refinement_iters;
};

struct swizzlepattern {
    uint8_t r;
    uint8_t g;
    uint8_t b;
    uint8_t a;
};

typedef uint16_t sf16;
typedef uint32_t sf32;

enum roundmode {
    SF_UP = 0,
    SF_DOWN = 1,
    SF_TOZERO = 2,
    SF_NEARESTEVEN = 3,
    SF_NEARESTAWAY = 4
};

// ---------------------------------------------------------------------------------------------
// Declare functions that are implemented in astcenc
// ---------------------------------------------------------------------------------------------

extern void encode_astc_image(
    const astc_codec_image* input_image,
    astc_codec_image* output_image,
    int xdim,
    int ydim,
    int zdim,
    const error_weighting_params* ewp,
    astc_decode_mode decode_mode,
    swizzlepattern swz_encode,
    swizzlepattern swz_decode,
    uint8_t* buffer,
    int pack_and_unpack,
    int threadcount);

extern void expand_block_artifact_suppression(
    int xdim,
    int ydim,
    int zdim,
    error_weighting_params* ewp);

extern void find_closest_blockdim_2d(
    float target_bitrate,
    int *x,
    int *y,
    int consider_illegal);

extern void find_closest_blockdim_3d(
    float target_bitrate,
    int *x,
    int *y,
    int *z,
    int consider_illegal);

extern void destroy_image(astc_codec_image* img);

extern astc_codec_image* allocate_image(int bitness, int xsize, int ysize, int zsize, int padding);

extern void test_inappropriate_extended_precision();
extern void prepare_angular_tables();
extern void build_quantization_mode_table();

extern "C" {
    sf16 float_to_sf16(float, roundmode);
}

// ---------------------------------------------------------------------------------------------
// Implement functions that implement our public-facing API
// ---------------------------------------------------------------------------------------------

namespace image {

AstcTexture astcCompress(const LinearImage& source, AstcConfig config) {

    // ---------------------------------------------------------------------------------------------
    // If this is the first time, initialize the ARM encoder tables.
    // ---------------------------------------------------------------------------------------------

    static bool first = true;
    if (first) {
        test_inappropriate_extended_precision();
        prepare_angular_tables();
        build_quantization_mode_table();
        first = false;
    }

    // ---------------------------------------------------------------------------------------------
    // Create an input image for the ARM encoder in a format that it can consume.
    // The encoder can take half-floats or bytes, but we always give it half-floats.
    // ---------------------------------------------------------------------------------------------

    const uint32_t width = source.getWidth();
    const uint32_t height = source.getHeight();
    astc_codec_image* input_image = allocate_image(16, width, height, 1, 0);
    constexpr bool flip = false;
    auto imagedata16 = input_image->imagedata16[0];
    for (int y = 0; y < height; y++) {
        int y_dst = y;
        int y_src = flip ? (height - y - 1) : y;
        float const* src = source.getPixelRef(0, y);
        for (int x = 0; x < width; x++) {
            imagedata16[y_dst][4 * x] = float_to_sf16(src[4 * x], SF_NEARESTEVEN);
            imagedata16[y_dst][4 * x + 1] = float_to_sf16(src[4 * x + 1], SF_NEARESTEVEN);
            imagedata16[y_dst][4 * x + 2] = float_to_sf16(src[4 * x + 2], SF_NEARESTEVEN);
            imagedata16[y_dst][4 * x + 3] = float_to_sf16(src[4 * x + 3], SF_NEARESTEVEN);
        }
    }

    // ---------------------------------------------------------------------------------------------
    // Determine the block size based on the bit rate.
    // ---------------------------------------------------------------------------------------------

    int xdim_2d, ydim_2d;
    int xdim_3d, ydim_3d, zdim_3d;
    find_closest_blockdim_2d(config.bitrate, &xdim_2d, &ydim_2d, 0);
    find_closest_blockdim_3d(config.bitrate, &xdim_3d, &ydim_3d, &zdim_3d, 0);
    const float log10_texels_2d = std::log((float)(xdim_2d * ydim_2d)) / std::log(10.0f);
    const float log10_texels_3d = std::log((float)(xdim_3d * ydim_3d * zdim_3d)) / log(10.0f);

    // ---------------------------------------------------------------------------------------------
    // Set up presets
    // ---------------------------------------------------------------------------------------------

    int plimit_autoset;
    float oplimit_autoset;
    float dblimit_autoset_2d;
    float dblimit_autoset_3d;
    float bmc_autoset;
    float mincorrel_autoset;
    int maxiters_autoset;
    int pcdiv;

    switch (config.quality) {
        // TODO: honor the other presets
        default:
        case AstcPreset::VERYFAST:
            plimit_autoset = 2;
            oplimit_autoset = 1.0;
            dblimit_autoset_2d = std::max(70 - 35 * log10_texels_2d, 53 - 19 * log10_texels_2d);
            dblimit_autoset_3d = std::max(70 - 35 * log10_texels_3d, 53 - 19 * log10_texels_3d);
            bmc_autoset = 25;
            mincorrel_autoset = 0.5;
            maxiters_autoset = 1;
            switch (ydim_2d) {
                case 4: pcdiv = 240; break;
                case 5: pcdiv = 56; break;
                case 6: pcdiv = 64; break;
                case 8: pcdiv = 47; break;
                case 10: pcdiv = 36; break;
                case 12: pcdiv = 30; break;
                default: pcdiv = 30; break;
            }
            break;
    }

    if (plimit_autoset < 1) {
        plimit_autoset = 1;
    } else if (plimit_autoset > PARTITION_COUNT) {
        plimit_autoset = PARTITION_COUNT;
    }

    error_weighting_params ewp;
    ewp.rgb_power = 1.0f;
    ewp.alpha_power = 1.0f;
    ewp.rgb_base_weight = 1.0f;
    ewp.alpha_base_weight = 1.0f;
    ewp.rgb_mean_weight = 0.0f;
    ewp.rgb_stdev_weight = 0.0f;
    ewp.alpha_mean_weight = 0.0f;
    ewp.alpha_stdev_weight = 0.0f;
    ewp.rgb_mean_and_stdev_mixing = 0.0f;
    ewp.mean_stdev_radius = 0;
    ewp.enable_rgb_scale_with_alpha = 0;
    ewp.alpha_radius = 0;
    ewp.block_artifact_suppression = 0.0f;
    ewp.rgba_weights[0] = 1.0f;
    ewp.rgba_weights[1] = 1.0f;
    ewp.rgba_weights[2] = 1.0f;
    ewp.rgba_weights[3] = 1.0f;
    ewp.ra_normal_angular_scale = 0;
    ewp.max_refinement_iters = maxiters_autoset;
    ewp.block_mode_cutoff = bmc_autoset / 100.0f;
    ewp.texel_avg_error_limit = pow(0.1f, dblimit_autoset_2d * 0.1f) * 65535.0f * 65535.0f;
    ewp.partition_1_to_2_limit = oplimit_autoset;
    ewp.lowest_correlation_cutoff = mincorrel_autoset;
    ewp.partition_search_limit = plimit_autoset;

    // For now we do not support 3D textures but we keep the variable names consistent
    // with what's found in the ARM standalone tool.
    int xdim = xdim_2d, ydim = ydim_2d, zdim = 1;
    expand_block_artifact_suppression(xdim, ydim, zdim, &ewp);

    // ---------------------------------------------------------------------------------------------
    // To help with debugging, dump the encoding settings in a format similar to the astcenc tool
    // ---------------------------------------------------------------------------------------------

    printf("2D Block size: %dx%d (%.2f bpp)\n", xdim_2d, ydim_2d, 128.0 / (xdim_2d * ydim_2d));
    printf("Max refinement iterations: %d (%s)\n", ewp.max_refinement_iters, "preset");
    #ifdef VERBOSE
    printf("Encoding settings:\n\n");
    printf("Target bitrate provided: %.2f bpp\n", config.bitrate);
    printf("3D Block size: %dx%dx%d (%.2f bpp)\n", xdim_3d, ydim_3d, zdim_3d, 128.0 / (xdim_3d * ydim_3d * zdim_3d));
    printf("Radius for mean-and-stdev calculations: %d texels\n", ewp.mean_stdev_radius);
    printf("RGB power: %g\n", ewp.rgb_power);
    printf("RGB base-weight: %g\n", ewp.rgb_base_weight);
    printf("RGB local-mean weight: %g\n", ewp.rgb_mean_weight);
    printf("RGB local-stdev weight: %g\n", ewp.rgb_stdev_weight);
    printf("RGB mean-and-stdev mixing across color channels: %g\n", ewp.rgb_mean_and_stdev_mixing);
    printf("Alpha power: %g\n", ewp.alpha_power);
    printf("Alpha base-weight: %g\n", ewp.alpha_base_weight);
    printf("Alpha local-mean weight: %g\n", ewp.alpha_mean_weight);
    printf("Alpha local-stdev weight: %g\n", ewp.alpha_stdev_weight);
    printf("RGB weights scale with alpha: ");
    if (ewp.enable_rgb_scale_with_alpha)
        printf("enabled (radius=%d)\n", ewp.alpha_radius);
    else
        printf("disabled\n");
    printf("Color channel relative weighting: R=%g G=%g B=%g A=%g\n",
            ewp.rgba_weights[0], ewp.rgba_weights[1], ewp.rgba_weights[2], ewp.rgba_weights[3]);
    printf("Block-artifact suppression parameter : %g\n", ewp.block_artifact_suppression);
    printf("Number of distinct partitionings to test: %d (%s)\n", ewp.partition_search_limit, "preset");
    printf("PSNR decibel limit: 2D: %f 3D: %f (%s)\n", dblimit_autoset_2d, dblimit_autoset_3d, "preset");
    printf("1->2 partition limit: %f\n", oplimit_autoset);
    printf("Dual-plane color-correlation cutoff: %f (%s)\n", mincorrel_autoset, "preset");
    printf("Block Mode Percentile Cutoff: %f (%s)\n", ewp.block_mode_cutoff * 100.0f, "preset");
    #endif

    // ---------------------------------------------------------------------------------------------
    // Perform compression
    // ---------------------------------------------------------------------------------------------

    constexpr int threadcount = 1; // TODO: set this thread count
    constexpr astc_decode_mode decode_mode = DECODE_LDR; // TODO: honor the config semantic
    constexpr swizzlepattern swz_encode = { 0, 1, 2, 3 };
    constexpr swizzlepattern swz_decode = { 0, 1, 2, 3 };

    const int xsize = input_image->xsize;
    const int ysize = input_image->ysize;
    const int zsize = input_image->zsize;
    const int xblocks = (xsize + xdim - 1) / xdim;
    const int yblocks = (ysize + ydim - 1) / ydim;
    const int zblocks = (zsize + zdim - 1) / zdim;

    uint32_t size = xblocks * yblocks * zblocks * 16;
    uint8_t* buffer = new uint8_t[size];

    encode_astc_image(input_image, nullptr, xdim, ydim, zdim, &ewp, decode_mode,
            swz_encode, swz_decode, buffer, 0, threadcount);

    destroy_image(input_image);

    return AstcTexture {
        .gl_internal_format = 0, // TODO: figure out the correct GL enum here
        .size = size,
        .data = decltype(AstcTexture::data)(buffer)
    };
}

AstcConfig astcParseOptionString(const string& configString) {
    const size_t _1 = configString.find('_');
    const size_t _2 = configString.find('_', _1 + 1);
    if (_1 == string::npos || _2 == string::npos) {
        return {};
    }
    string quality = configString.substr(0, _1);
    string semantic = configString.substr(_1 + 1, _2 - _1 - 1);
    string bitrate = configString.substr(_2 + 1);
    AstcConfig config;
    if (quality == "veryfast") {
        config.quality = AstcPreset::VERYFAST;
    } else if (quality == "fast") {
        config.quality = AstcPreset::FAST;
    } else if (quality == "medium") {
        config.quality = AstcPreset::MEDIUM;
    } else if (quality == "thorough") {
        config.quality = AstcPreset::THOROUGH;
    } else if (quality == "exhaustive") {
        config.quality = AstcPreset::EXHAUSTIVE;
    } else {
        return {};
    }
    if (semantic == "ldr") {
        config.semantic = AstcSemantic::COLORS_LDR;
    } else if (semantic == "hdr") {
        config.semantic = AstcSemantic::COLORS_HDR;
    } else if (semantic == "normals") {
        config.semantic = AstcSemantic::NORMALS;
    } else {
        return {};
    }
    config.bitrate = std::atof(bitrate.c_str());
    return config;
}

} // namespace image
