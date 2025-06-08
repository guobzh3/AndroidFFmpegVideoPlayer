precision mediump float;

varying vec2 v_coord;

// --- UNIFORMS ---
// Textures
uniform sampler2D u_BaseTexture;      // Pass 1: Input video frame. Pass 2: Upscaled background texture.
uniform sampler2D u_SrPatchTexture;   // Pass 2: The SR patch texture.

// Parameters
uniform vec2      u_TexelSize;        // Pass 1: Texel size for the input video frame.
uniform vec4      u_PatchRect;        // Pass 2: Patch rectangle in normalized coordinates [x, y, width, height].
uniform bool      u_DrawPatch;        // Pass 2: Flag to control if the patch is drawn.

// Mode Control
uniform int       u_RenderMode;       // 0 for Upscale Pass, 1 for Composite Pass.

// --- BILINEAR UPSCALE FUNCTION (for Pass 1) ---
vec4 textureBilinear(sampler2D tex, vec2 texCoords, vec2 texelSize) {
    vec2 first_texel_int_coord = floor(texCoords / texelSize);
    vec2 fraction = fract(texCoords / texelSize);

    vec2 uv_bottom_left  = (first_texel_int_coord + vec2(0.0, 0.0)) * texelSize;
    vec2 uv_bottom_right = (first_texel_int_coord + vec2(1.0, 0.0)) * texelSize;
    vec2 uv_top_left     = (first_texel_int_coord + vec2(0.0, 1.0)) * texelSize;
    vec2 uv_top_right    = (first_texel_int_coord + vec2(1.0, 1.0)) * texelSize;

    vec4 c00 = texture2D(tex, uv_bottom_left);
    vec4 c10 = texture2D(tex, uv_bottom_right);
    vec4 c01 = texture2D(tex, uv_top_left);
    vec4 c11 = texture2D(tex, uv_top_right);

    vec4 tx0 = mix(c00, c10, fraction.x);
    vec4 tx1 = mix(c01, c11, fraction.x);

    return mix(tx0, tx1, fraction.y);
}
// --- CUBIC INTERPOLATION HELPER FUNCTION ---
// Implements the Catmull-Rom cubic spline interpolation for a single dimension.
// It calculates an interpolated value between p1 and p2, using p0 and p3 as control points
// to define the gradient at the boundaries.
// @param p0, p1, p2, p3: The four consecutive sample points (e.g., colors).
// @param t: The fractional distance between p1 and p2 (from 0.0 to 1.0).
vec4 cubic(vec4 p0, vec4 p1, vec4 p2, vec4 p3, float t) {
    // The polynomial is of the form: a*t^3 + b*t^2 + c*t + d
    // The coefficients are derived from the Catmull-Rom spline constraints.
    vec4 a = (3.0 * p1 - 3.0 * p2 + p3 - p0) * 0.5;
    vec4 b = (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * 0.5;
    vec4 c = (p2 - p0) * 0.5;
    vec4 d = p1;

    return t * t * t * a + t * t * b + t * c + d;
}

// --- BICUBIC UPSCALE FUNCTION ---
// Performs bicubic interpolation on a texture.
// @param tex: The source texture sampler.
// @param texCoords: The texture coordinates to sample at (in normalized 0-1 space).
// @param texelSize: The size of a single texel (1.0 / texture_resolution).
vec4 textureBicubic(sampler2D tex, vec2 texCoords, vec2 texelSize) {
    // Convert normalized texture coordinates to texel coordinates
    vec2 texel_coord = texCoords / texelSize;

    // Get the fractional part of the texel coordinate, which is our interpolation factor 't'
    vec2 f = fract(texel_coord);

    // Get the integer coordinate of the top-left texel in the 4x4 grid
    vec2 int_coord = floor(texel_coord) - 1.0;

    // Perform 4 horizontal cubic interpolations (one for each row of the 4x4 grid)
    vec4 H[4];
    for (int j = 0; j < 4; j++) {
        // Sample the four texels for the current row
        vec4 p0 = texture2D(tex, (int_coord + vec2(0.0, j)) * texelSize);
        vec4 p1 = texture2D(tex, (int_coord + vec2(1.0, j)) * texelSize);
        vec4 p2 = texture2D(tex, (int_coord + vec2(2.0, j)) * texelSize);
        vec4 p3 = texture2D(tex, (int_coord + vec2(3.0, j)) * texelSize);

        // Interpolate horizontally using the x-fraction
        H[j] = cubic(p0, p1, p2, p3, f.x);
    }

    // Perform 1 final vertical cubic interpolation on the results of the horizontal ones
    // using the y-fraction to get the final color.
    return cubic(H[0], H[1], H[2], H[3], f.y);
}

// --- MAIN LOGIC ---
void main() {
    if (u_RenderMode == 0) {
        // --- Pass 1: Upscale ---
        // Upscale the base texture using bilinear filtering.
        gl_FragColor = textureBicubic(u_BaseTexture, v_coord, u_TexelSize);

    } else {
        // --- Pass 2: Composite ---
        // By default, the color is from the upscaled background texture.
        vec4 finalColor = texture2D(u_BaseTexture, v_coord);

        // If the patch should be drawn and we are inside its rectangle...
        if (u_DrawPatch &&
            v_coord.x >= u_PatchRect.x && v_coord.x <= u_PatchRect.x + u_PatchRect.z &&
            v_coord.y >= u_PatchRect.y && v_coord.y <= u_PatchRect.y + u_PatchRect.w) {

            // Map the screen coordinate to the patch's own texture coordinate.
            vec2 patchCoord = (v_coord.xy - u_PatchRect.xy) / u_PatchRect.zw;
            finalColor = texture2D(u_SrPatchTexture, patchCoord);
        }

        gl_FragColor = finalColor;
    }
}
