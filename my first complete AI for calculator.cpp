#include <iostream> // 输入输出
#include <vector>   // 向量
#include <string>   // 字符串
#include <cmath> 
#include<vector>
#include<random>
#include<algorithm>
using namespace std;
int input_node,hidden_node,output_node;//每层节点数目及节点 
double a=0.05;//学习率
vector<double> input_i,hidden_i,output_i;//存每个节点的输入值；
vector<double> hidden_o,output_o;//存每个节点的输出值； 
vector<vector<double>> weight1;//输入层到隐藏层的权重；[i][j],i是横向(下一个节点对应上层节点）
//j为下层节点数 
vector<vector<double>> weight2;//隐藏层到输出层的权重； 
vector<double> error_hidden,error_output;//隐藏层与输出层的误差；
vector<double> target;
double loss;
random_device rd;
mt19937 gen(rd());
uniform_real_distribution<> dis(0.0,0.1);
//设置随机权重
double sigmoid(double input)
{
	return 1/(1.0+exp(-input));
}//sigmoid函数 

void init()
{
	input_node=2;
	hidden_node=100;
	output_node=1;
	//分配内存
	input_i.resize(input_node);
	hidden_i.resize(hidden_node);
	output_i.resize(output_node);
	hidden_o.resize(hidden_node);
	output_o.resize(output_node);
	error_hidden.resize(hidden_node);
	error_output.resize(output_node);
	target.resize(output_node);
	weight1.resize(input_node,vector<double>(hidden_node,0));
	weight2.resize(hidden_node,vector<double>(output_node,0));
	fill(input_i.begin(),input_i.end(),0);
	fill(hidden_i.begin(),hidden_i.end(),0);
	fill(output_i.begin(),output_i.end(),0);
	fill(hidden_o.begin(),hidden_o.end(),0);
	fill(output_o.begin(),output_o.end(),0);//将数组中的所有元素清零 
	fill(error_hidden.begin(),error_hidden.end(),0);
	fill(error_output.begin(),error_output.end(),0);
	for(int i = 0; i < input_node; i++) {
		    for(int j = 0; j < hidden_node; j++) {
		        weight1[i][j] = dis(gen); 
		    }
		}
		// 初始化 weight2
		for(int i = 0; i < hidden_node; i++) {
		    for(int j = 0; j < output_node; j++) {
		        weight2[i][j] = dis(gen);
		    }
		}
}
void _init_()//初始化函数，设置每层节点的数量 
{
	loss=0.0;
    fill(hidden_i.begin(), hidden_i.end(), 0.0);
    fill(output_i.begin(), output_i.end(), 0.0);
    fill(error_hidden.begin(), error_hidden.end(), 0.0);
	//隐藏层的输入值  矩阵
	for(int i=0;i<hidden_node;i++)//hidden_i[i]+=input_i[j]*weight1[j][i]
		for(int j=0;j<input_node;j++)
			hidden_i[i]+=input_i[j]*weight1[j][i]; 
	//计算隐藏层每个节点的输出值
	for(int i=0;i<hidden_node;i++)
		hidden_o[i]=sigmoid(hidden_i[i]); 
	//输出层的输入值  矩阵
	for(int i=0;i<output_node;i++)
		for(int j=0;j<hidden_node;j++)
			output_i[i]+=hidden_o[j]*weight2[j][i];
	//计算输出层每个节点的输出值 
	for(int i=0;i<output_node;i++)
		output_o[i]=sigmoid(output_i[i]);
	//计算输出值与正常值的误差
	for(int i=0;i<output_node;i++)
		error_output[i]=target[i]-output_o[i];
	//计算loss
	for(int i=0;i<output_node;i++)
		loss+=error_output[i]*error_output[i];
	loss=loss/(output_node);
	//反向传播误差
	//计算输出层每个节点的总和权重
	vector<double> weight_sum(output_node, 0);
	for(int i=0;i<output_node;i++)
		for(int j=0;j<hidden_node;j++)
			weight_sum[i]+=weight2[j][i];
	//计算隐藏层每个节点的误差
	for(int i=0;i<hidden_node;i++)
		for(int j=0;j<output_node;j++)
			error_hidden[i]+=error_output[j]*(weight2[i][j]/(weight_sum[j]+0.0000001));
	//更新权重
	//输入层到隐藏层的权重；[i][j],i是横向(下一个节点对应上层节点）
	//j为下层节点数 
	for(int i=0;i<hidden_node;i++)
		for(int j=0;j<output_node;j++)
		{
			double h2=weight2[i][j];
			weight2[i][j]=h2-a*((-error_output[j])*sigmoid(output_i[j])*(1-sigmoid(output_i[j]))*hidden_o[i]);
		}
	for(int i=0;i<input_node;i++)
		for(int j=0;j<hidden_node;j++)
		{
			double h1=weight1[i][j];
			weight1[i][j]=h1-a*((-error_hidden[j])*sigmoid(hidden_i[j])*(1-sigmoid(hidden_i[j]))*input_i[i]);
		}
}

void train()//数据 训练 
{
	vector<vector<double>> train_x={
		{0.0, 0.0}, {0.0, 0.1}, {0.0, 0.2}, {0.0, 0.3}, {0.0, 0.4},
        {0.1, 0.0}, {0.1, 0.1}, {0.1, 0.2}, {0.1, 0.3}, {0.1, 0.4},
        {0.2, 0.0}, {0.2, 0.1}, {0.2, 0.2}, {0.2, 0.3}, {0.2, 0.4},
        {0.3, 0.0}, {0.3, 0.1}, {0.3, 0.2}, {0.3, 0.3}, {0.3, 0.4},
        {0.4, 0.0}, {0.4, 0.1}, {0.4, 0.2}, {0.4, 0.3}, {0.4, 0.4},
        {0.0, 0.0}, {0.0, 0.1}, {0.0, 0.2}, {0.0, 0.3}, {0.0, 0.4},
	};
	vector<double> train_y={
		0.0, 0.1, 0.2, 0.3, 0.4,
        0.1, 0.2, 0.3, 0.4, 0.5,
        0.2, 0.3, 0.4, 0.5, 0.6,
        0.3, 0.4, 0.5, 0.6, 0.7,
        0.4, 0.5, 0.6, 0.7, 0.8,
        0.0, 0.1, 0.2, 0.3, 0.4,
	};
	double total_loss=0.0;
	for(int epoch=0;epoch<=100000;epoch++)
	{
		total_loss=0.0;
		for(size_t i=0;i<train_x.size();i++)
			{
				input_i[0]=train_x[i][0];
				input_i[1]=train_x[i][1];
				target[0]=train_y[i];
				_init_();
				total_loss+=loss;
			}
		if(epoch%100==0)
			cout<<"epoch"<<epoch<<"--loss:"<<total_loss/train_x.size()<<endl;
	}
		
}
void query()//给定然后进行输出 
{
	
}
int main()
{
	init();
	train();
	cin>>input_i[0];
	cin>>input_i[1];
	target[0]=input_i[0]+input_i[1];
	_init_();
	cout<<output_o[0];
	system("pause");
	return 0;
}