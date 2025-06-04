precision mediump float;

varying vec2 v_coord; // 输入的纹理坐标 (0.0 to 1.0)
uniform sampler2D u_Texture; // 输入纹理
uniform vec2 u_TexelSize;   // 输入纹理的单个纹素大小 (1.0/textureWidth, 1.0/textureHeight)

// 双三次权重函数
// Catmull-Rom spline variant (a = 0.5)
// w(x, a) = { (a+2)|x|^3 - (a+3)|x|^2 + 1,     for |x| <= 1
//           { a|x|^3 - 5a|x|^2 + 8a|x| - 4a,  for 1 < |x| <= 2
//           { 0,                               otherwise
// For bicubic, a typical choice is a = -0.5 or a = -0.75.
// Here we use a simplified version or a common one.
// This is a common cubic Hermite spline interpolation kernel.
vec4 cubic(float v) {
    vec4 n = vec4(1.0, 2.0, 3.0, 4.0) - v;
    vec4 s = n * n * n;
    float x = s.x;
    float y = s.y - 4.0 * s.x;
    float z = s.z - 4.0 * s.y + 6.0 * s.x;
    float w = 6.0 - x - y - z; // s.w - 4.0 * s.z + 6.0 * s.y - 4.0 * s.x (but s.w is (4-v)^3 which is not what we want)
                               // The sum of weights must be 1. For Catmull-Rom, the weights are:
                               // W_0 = (-t^3 + 2t^2 - t)/2
                               // W_1 = (3t^3 - 5t^2 + 2)/2
                               // W_2 = (-3t^3 + 4t^2 + t)/2
                               // W_3 = (t^3 - t^2)/2
                               // This cubic function is a common one for interpolation.
    return vec4(x, y, z, w) * (1.0/6.0);
}

vec4 textureBicubic(sampler2D tex, vec2 texCoords) {
    vec2 f = fract(texCoords / u_TexelSize - 0.5); // Fractional part of the coordinate, adjusted by 0.5
    vec2 p0 = (floor(texCoords / u_TexelSize - 0.5) - 1.0) * u_TexelSize; // Center of the 4x4 neighborhood, p(-1)
    vec2 p1 = (floor(texCoords / u_TexelSize - 0.5) + 0.0) * u_TexelSize; // p(0)
    vec2 p2 = (floor(texCoords / u_TexelSize - 0.5) + 1.0) * u_TexelSize; // p(1)
    vec2 p3 = (floor(texCoords / u_TexelSize - 0.5) + 2.0) * u_TexelSize; // p(2)

    // Calculate the cubic weights
    vec4 w0 = cubic(f.x);
    vec4 w1 = cubic(f.y);

    // Sample the 16 surrounding texels
    vec4 c00 = texture2D(tex, vec2(p0.x, p0.y));
    vec4 c10 = texture2D(tex, vec2(p1.x, p0.y));
    vec4 c20 = texture2D(tex, vec2(p2.x, p0.y));
    vec4 c30 = texture2D(tex, vec2(p3.x, p0.y));

    vec4 c01 = texture2D(tex, vec2(p0.x, p1.y));
    vec4 c11 = texture2D(tex, vec2(p1.x, p1.y));
    vec4 c21 = texture2D(tex, vec2(p2.x, p1.y));
    vec4 c31 = texture2D(tex, vec2(p3.x, p1.y));

    vec4 c02 = texture2D(tex, vec2(p0.x, p2.y));
    vec4 c12 = texture2D(tex, vec2(p1.x, p2.y));
    vec4 c22 = texture2D(tex, vec2(p2.x, p2.y));
    vec4 c32 = texture2D(tex, vec2(p3.x, p2.y));

    vec4 c03 = texture2D(tex, vec2(p0.x, p3.y));
    vec4 c13 = texture2D(tex, vec2(p1.x, p3.y));
    vec4 c23 = texture2D(tex, vec2(p2.x, p3.y));
    vec4 c33 = texture2D(tex, vec2(p3.x, p3.y));

    // Interpolate along x-axis
    vec4 r0 = c00 * w0.x + c10 * w0.y + c20 * w0.z + c30 * w0.w;
    vec4 r1 = c01 * w0.x + c11 * w0.y + c21 * w0.z + c31 * w0.w;
    vec4 r2 = c02 * w0.x + c12 * w0.y + c22 * w0.z + c32 * w0.w;
    vec4 r3 = c03 * w0.x + c13 * w0.y + c23 * w0.z + c33 * w0.w;

    // Interpolate along y-axis
    return r0 * w1.x + r1 * w1.y + r2 * w1.z + r3 * w1.w;
}

vec4 textureBilinear(sampler2D tex, vec2 texCoords) {

    vec2 pixel_coords = texCoords / u_TexelSize - 0.5;

    vec2 p0_idx = floor(pixel_coords);


    vec2 fract_offset = fract(pixel_coords); // 等价于 pixel_coords - p0_idx

    vec2 uv00 = (p0_idx + vec2(0.0, 0.0)) * u_TexelSize;
    vec2 uv10 = (p0_idx + vec2(1.0, 0.0)) * u_TexelSize;
    vec2 uv01 = (p0_idx + vec2(0.0, 1.0)) * u_TexelSize;
    vec2 uv11 = (p0_idx + vec2(1.0, 1.0)) * u_TexelSize;
    

    vec2 first_texel_int_coord = floor(texCoords / u_TexelSize);
    vec2 fraction = fract(texCoords / u_TexelSize);

    vec2 uv_bottom_left  = (first_texel_int_coord + vec2(0.0, 0.0)) * u_TexelSize;
    vec2 uv_bottom_right = (first_texel_int_coord + vec2(1.0, 0.0)) * u_TexelSize;
    vec2 uv_top_left     = (first_texel_int_coord + vec2(0.0, 1.0)) * u_TexelSize;
    vec2 uv_top_right    = (first_texel_int_coord + vec2(1.0, 1.0)) * u_TexelSize;


    vec4 c00 = texture2D(tex, uv_bottom_left);  
    vec4 c10 = texture2D(tex, uv_bottom_right); 
    vec4 c01 = texture2D(tex, uv_top_left);     
    vec4 c11 = texture2D(tex, uv_top_right);    

    vec4 tx0 = mix(c00, c10, fraction.x); 
    vec4 tx1 = mix(c01, c11, fraction.x); 

    return mix(tx0, tx1, fraction.y); // 最终颜色
}

void main() {
    // u_TexelSize should be passed from Java: vec2(1.0/textureWidth, 1.0/textureHeight)
    // of the *input* texture.
    gl_FragColor = textureBilinear(u_Texture, v_coord);
}
