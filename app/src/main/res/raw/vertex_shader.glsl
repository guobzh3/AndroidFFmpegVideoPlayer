// attribute是从java 程序中传进来的参数

attribute vec2 a_position;
attribute vec2 a_coord ;

varying vec2 v_coord;

//uniform v_matrix;

//
void main(){
    // 设置顶点
    gl_Position= vec4(a_position,  0.0,1.0);

    // 传递纹理坐标给fragemt_shader
    v_coord = a_coord;
}
