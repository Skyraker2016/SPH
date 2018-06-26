// CUDA runtime 库 + CUBLAS 库   
#include "cuda_runtime.h"  
#include "cublas_v2.h"  
#include<device_launch_parameters.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glut.h>
#include <vector>
#include <iostream>
#include <cmath>
using namespace std;

//网格边界
#define LEDGE -40
#define REDGE 40
#define LEVEL 50
#define LSIDE -40
#define RSIDE 40
#define FSIDE 40
#define BSIDE -40
#define USIDE 40
#define DSIDE -40
//点数量
#define xnum 16
#define ynum 16
#define znum 30


#define G -70.0
#define H 2.0
#define P0 1.0
#define K 50.0
#define PI 3.14159
#define M 1.0
#define U 200.0

#define BALL 3
#define PP 0.1

#define InitStep 1


#define t 0.1;


//typedef struct {
//	double x, y, z;
//	double vx, vy, vz;
//	double ax, ay, az;
//	double d;
//	double p;
//} Point;


void display();
void draw_point();

#define WIDTH 800
#define HEIGHT 800

float p_x[xnum*ynum*znum];
float p_y[xnum*ynum*znum];
float p_z[xnum*ynum*znum];
//float p_vx[xnum*ynum*znum];
//float p_vy[xnum*ynum*znum];
//float p_vz[xnum*ynum*znum];
//float p_ax[xnum*ynum*znum];
//float p_ay[xnum*ynum*znum];
//float p_az[xnum*ynum*znum];
float p_d[xnum*ynum*znum];
//float p_p[xnum*ynum*znum];

__global__ void kenelInitPoint(float *x, float *y, float *z, float *vx, float *vy, float *vz)
{

	int idx = threadIdx.x + blockIdx.x*ynum*znum;
	int i = idx / (ynum * znum);
	int j = idx / znum % ynum;
	int k = idx % znum;
	x[idx] = -InitStep*xnum / 2 + InitStep * i;
	y[idx] = -InitStep * xnum / 2 + InitStep * j;
	z[idx] = -InitStep * xnum / 2 + InitStep * k - 30;
	vx[idx] = 0;
	vy[idx] = 0;
	vz[idx] = 0;
	//printf(" xx %lf", z[idx]);
}


__global__ void kernelCountPoint_1(float *x, float *y, float *z, float *d, float *p) {
	int idx = threadIdx.x + blockIdx.x*ynum*znum;
	d[idx] = 0;
	for (int i = 0; i < xnum*ynum*znum; i++) {
		float dd = (x[idx] - x[i])*(x[idx] - x[i]) + (y[idx] - y[i])*(y[idx] - y[i]) + (z[idx] - z[i])*(z[idx] - z[i]);
		if (dd<H*H)
		{
			d[idx] += M * 315 * (H*H - dd)*(H*H - dd)*(H*H - dd) / (64 * PI*H*H*H*H*H*H*H*H*H);
		}
	}
	p[idx] = K * (d[idx] - P0);
}
__global__ void kernelCountPoint_2(float *x, float *y, float *z, float *vx, float *vy, float *vz, float *p, float *ax, float *ay, float *az) {
	int idx = threadIdx.x + blockIdx.x*ynum*znum;
	double yalix = 0, niandux = 0, yaliy = 0, nianduy = 0, yaliz = 0, nianduz = 0;
	for (int i = 0; i < xnum*ynum*znum; i++) {
		if (i != idx) {
			float d = (x[idx] - x[i])*(x[idx] - x[i]) + (y[idx] - y[i])*(y[idx] - y[i]) + (z[idx] - z[i])*(z[idx] - z[i]);
			if (d<H*H)
			{
				yaliz += (((p[idx] +p[i])* (z[idx] - z[i]) * ((H - sqrt(d))*(H - sqrt(d))) / (sqrt(d)  * (2 * p[idx]*p[i]))));
				nianduz += (vz[i] - vz[idx])  * (H - sqrt(d)) / (p[idx]*p[i]);

				yalix += (((p[idx] +p[i])* (x[idx] - x[i]) * ((H - sqrt(d))*(H - sqrt(d))) / (sqrt(d)  * (2 * p[idx]*p[i]))));
				niandux += (vx[i] - vx[idx])  * (H - sqrt(d)) / (p[idx]*p[i]);

				yaliy += (((p[idx] +p[i])* (y[idx] - y[i]) * ((H - sqrt(d))*(H - sqrt(d))) / (sqrt(d)  * (2 * p[idx]*p[i]))));
				nianduy += (vy[i] - vy[idx])  * (H - sqrt(d)) / (p[idx]*p[i]);

			}
		}
	}
	ax[idx] += 0.5 * M * 45 * yalix / (PI*H*H*H*H*H*H) + 0.2 * M * 45 * U*niandux / (PI*H*H*H*H*H*H);
	ay[idx] += 0.5 * M * 45 * yaliy / (PI*H*H*H*H*H*H) + 0.2 * M * 45 * U*nianduy / (PI*H*H*H*H*H*H);
	az[idx] += 0.5 * M * 45 * yaliz / (PI*H*H*H*H*H*H) + 0.2 * M * 45 * U*nianduz / (PI*H*H*H*H*H*H);
}
__global__ void kernelCountPoint_3(float *x, float *y, float *z, float *vx, float *vy, float *vz, float *p, float *ax, float *ay, float *az) {
	int idx = threadIdx.x + blockIdx.x*ynum*znum;
	double aax = ax[idx];
	double aay = ay[idx];
	double aaz = az[idx];
	aaz += G;
	vx[idx] = vx[idx] + aax * t;
	vy[idx] = vy[idx] + aay * t;
	vz[idx] = vz[idx] + aaz * t;


	x[idx] += vx[idx]*t;
	y[idx] += vy[idx]*t;
	z[idx] += vz[idx]*t;

	if (z[idx] < DSIDE ) {
		z[idx] = DSIDE;
		az[idx] = vz[idx] * (-2);
		vz[idx] = vz[idx] + aaz * t;
	}
	if (x[idx] < BSIDE || x[idx]>FSIDE) {
		x[idx] = x[idx] < BSIDE ? BSIDE : FSIDE;
		ax[idx] = vx[idx] * (-2);
		vx[idx] = vx[idx] + aax * t;
	}
	if (y[idx] < LSIDE || y[idx]>RSIDE) {
		y[idx] = y[idx] < LSIDE ? LSIDE : RSIDE;
		ay[idx] = vy[idx] * (-2);
		vy[idx] = vy[idx] + aay * t;
	}
}
float kd = 0;

int main()
{
	//int num = 0;
	//cudaDeviceProp prop;
	//cudaGetDeviceCount(&num);
	//for (int i = 0; i<num; i++)
	//{
	//	cudaGetDeviceProperties(&prop, i);
	//}
	glfwInit();
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

	//初始化窗口
	GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "fluid", NULL, NULL);
	glfwMakeContextCurrent(window);


	//初始化glew
	glewExperimental = GL_TRUE;
	glewInit();

	//激活深度测试
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);


	float *dev_x = 0;
	float *dev_y = 0;
	float *dev_z = 0;
	float *dev_vx = 0;
	float *dev_vy = 0;
	float *dev_vz = 0;
	float *dev_ax = 0;
	float *dev_ay = 0;
	float *dev_az = 0;
	float *dev_d = 0;
	float *dev_p = 0;
	cudaMalloc((void**)&dev_x, xnum*ynum*znum * sizeof(float));
	cudaMalloc((void**)&dev_y, xnum*ynum*znum * sizeof(float));
	cudaMalloc((void**)&dev_z, xnum*ynum*znum * sizeof(float));
	cudaMalloc((void**)&dev_vx, xnum*ynum*znum * sizeof(float));
	cudaMalloc((void**)&dev_vy, xnum*ynum*znum * sizeof(float));
	cudaMalloc((void**)&dev_vz, xnum*ynum*znum * sizeof(float));
	cudaMalloc((void**)&dev_ax, xnum*ynum*znum * sizeof(float));
	cudaMalloc((void**)&dev_ay, xnum*ynum*znum * sizeof(float));
	cudaMalloc((void**)&dev_az, xnum*ynum*znum * sizeof(float));
	cudaMalloc((void**)&dev_d, xnum*ynum*znum * sizeof(float));
	cudaMalloc((void**)&dev_p, xnum*ynum*znum * sizeof(float));
	kenelInitPoint << < xnum, ynum*znum >> > (dev_x, dev_y, dev_z, dev_vx, dev_vy, dev_vz);
	//背景色设置
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	while (!glfwWindowShouldClose(window)) {
		//cout << "working..." << endl;
		kernelCountPoint_1 << < xnum, ynum*znum >> > (dev_x, dev_y, dev_z, dev_d, dev_p);
		kernelCountPoint_2 << < xnum, ynum*znum >> > (dev_x, dev_y, dev_z, dev_vx, dev_vy, dev_vz, dev_p, dev_ax, dev_ay, dev_az );
		kernelCountPoint_3 << < xnum, ynum*znum >> > (dev_x, dev_y, dev_z, dev_vx, dev_vy, dev_vz, dev_p, dev_ax, dev_ay, dev_az);
		cudaMemcpy(p_x, dev_x, xnum*ynum*znum * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(p_y, dev_y, xnum*ynum*znum * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(p_z, dev_z, xnum*ynum*znum * sizeof(float), cudaMemcpyDeviceToHost);
		display();
		glfwSwapBuffers(window);
		glfwPollEvents();
	}
	glfwTerminate();
	cudaFree(dev_p);
	return 0;

}

//cudaError_t initPoint(Point * dev_p) {
//	cudaMemcpy(dev_p, point, xnum*ynum*znum * sizeof(Point), cudaMemcpyHostToDevice);
//	kenelInitPoint <<<xnum, ynum*znum >>> (dev_p);
//	cudaThreadSynchronize();
//	cudaMemcpy(point, dev_p, xnum*ynum*znum * sizeof(Point), cudaMemcpyDeviceToHost);
//	return cudaSuccess;
//}
//cudaError_t countPoint(Point * dev_p) {
//
//	cudaMemcpy(dev_p, point, xnum*ynum*znum * sizeof(Point), cudaMemcpyHostToDevice);
//
//	kernelCountPoint_1 << <xnum, ynum*znum >> > (dev_p);
//	kernelCountPoint_2 << <xnum, ynum*znum >> > (dev_p);
//	kernelCountPoint_3 << <xnum, ynum*znum >> > (dev_p);
//
//	cudaMemcpy(point, dev_p, xnum*ynum*znum * sizeof(Point), cudaMemcpyDeviceToHost);
//	return cudaSuccess;
//}

void display(){
    //清缓存
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    //加载单位矩阵
    glLoadIdentity();
    //透视矩阵
    gluPerspective(60.0, 1, REDGE, LEDGE);
    //摄像矩阵

     gluLookAt(150, 0, 0, 0, 0, 0, 0, 0, 1);

    //根据模式来进行绘制
    glColor3f(0.9, 0.6, 0.6);
    glBegin(GL_TRIANGLES);
    glVertex3f(LEDGE,LEDGE,LEDGE);
    glVertex3f(LEDGE,REDGE,LEDGE);
    glVertex3f(REDGE,REDGE,LEDGE);
        
    glVertex3f(REDGE,REDGE,LEDGE);
    glVertex3f(REDGE,LEDGE,LEDGE);
    glVertex3f(LEDGE,LEDGE,LEDGE);
    glEnd();

    glColor3f(0.0f, 0.0f, 0.0f);
    draw_point();
    //glColor3f(0.5f, 0.7f, 1.0f);
    //draw_face();
}
//
//void init(Point *dev_p){
//	initPoint(dev_p);
//	kd = (float)315 / (64 * PI*pow(H, 9));
//}

void draw_point(){
    for (int i=0; i<xnum; i++){
        for (int j=0; j<ynum; j++){
            for (int k=0; k<znum; k++){
				int idx = i * ynum*znum + j * znum + k;
				//cout << x[idx] << " " << y[idx] << " " << z[idx] << endl;
                glPointSize(2.0f);
                glBegin(GL_POINTS);
                glVertex3f(p_x[idx], p_y[idx], p_z[idx]);
                glEnd();
            }
        }
    }
}







