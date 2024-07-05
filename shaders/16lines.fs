
#version 450

layout(set=0, binding=0) uniform Globals {
    float diminish_red;
    float diminish_green;
    float diminish_blue;
} globals;

layout(location = 0) in float f_intensity;
layout(location = 1) in float f_index;
layout(location = 2) flat in int f_mode;
layout(location = 0) out vec4 f_color;

void main() {
    if(f_mode == -1){
        float red = min(f_intensity, 1);
        float green = min((red - 0.3) * 3, red);
        float blue = min((f_intensity - 0.9) * 10, red);
        f_color = vec4(
            red,
            green,
            blue,
            1
        );
    }else{
        if(f_intensity < 0.001)
            discard;
        float distance_to_center = pow(250/max(f_index, 1), 0.1);
        float distance_to_center_white = pow(250/max(f_index, 1), 0.1);
        float center = max(
                        max(
                                1 - abs(f_index - floor(f_index) - 0.545) * 2 / distance_to_center_white,
                                0
                        )
                        * (1/distance_to_center*10+1) - (1/distance_to_center*10 - 1),
                        0
                        )
                        * pow(f_intensity, 0.4)
                        ;
        if(center < 0.001)
            discard;
        float center_white = max(
                        max(
                                1 - abs(f_index - floor(f_index) - 0.525) * 2 / distance_to_center_white,
                                0
                        )
                        * (1/distance_to_center*10+1) - (1/distance_to_center*10 - 1),
                        0
                        )
                        * pow(f_intensity, 0.4)
                        ;
        if(f_mode == 0){
            distance_to_center = 1;
            center_white = 1;
            center = 1;
        }
        float red = min((pow(f_intensity, 0.3)*0.25 + f_intensity*0.75)/distance_to_center, 1) * center * globals.diminish_red;
        // red = 1/distance_to_center/100;
        f_color = vec4(
            // min(pow(f_intensity, 0.75)/distance_to_center*3, 1),
            // 1,
            red,
            min( (pow(f_intensity, 1)/distance_to_center*3 - 0.9) * globals.diminish_green, red),
            min( (pow(f_intensity, 1.2)/distance_to_center*pow(center_white, 0.5)*3 - 2) * globals.diminish_blue, red),
            1
            // 0.01 + min(f_intensity, red)*0.99/sqrt(distance_to_center)
            // min((pow(f_intensity, 0.5)/2 + f_intensity/2)/distance_to_center, 1) * center * 0.95
        );
    }
}