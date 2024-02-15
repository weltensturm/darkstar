
#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in float intensity;
layout(location = 2) in float index;
layout(location = 3) in int mode;
layout(location = 0) out float f_intensity;
layout(location = 1) out float f_index;
layout(location = 2) out int f_mode;

void main() {
    f_intensity = intensity;
    f_index = index;
    f_mode = mode;
    gl_Position = vec4(position, 1.0);
}
