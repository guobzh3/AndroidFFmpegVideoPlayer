precision mediump float;
varying vec2 v_coord;

// texture要如何传进来？
uniform sampler2D u_Texture;    // The input texture.

//bool isOutRect(vec2 coord) {
//    return coord.x > 1/4f && coord.x < 3/4f && coord.y > 1/4f && coord.y < 3/4f;
//}

void main (){
    gl_FragColor = texture2D(u_Texture , v_coord);
}

