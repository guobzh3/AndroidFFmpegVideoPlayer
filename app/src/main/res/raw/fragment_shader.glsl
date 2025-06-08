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

// --- MAIN LOGIC ---
void main() {
    if (u_RenderMode == 0) {
        // --- Pass 1: Upscale ---
        // Upscale the base texture using bilinear filtering.
        gl_FragColor = textureBilinear(u_BaseTexture, v_coord, u_TexelSize);

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
