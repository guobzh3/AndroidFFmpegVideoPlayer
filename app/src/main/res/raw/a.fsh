
attribute vec4 vposition; // 变量 float[4] ， 有几个点，着色器就会执行几次（gl中是4个象限表示一个点，几维就用几个点）
attribute vec2 vCord;
attribute vec2 aCord;

uniform mat4 vMatrix;

void main(){
    gl_position = vposition;
    aCord = vCord;
}