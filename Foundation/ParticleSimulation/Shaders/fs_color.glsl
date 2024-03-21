#version 440

uniform sampler2D tex;

in vec2 t;

layout(location = 0) out vec4 color;

void main()
{
	color = texture(tex, t);
}