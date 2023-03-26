
clc
clear
load('data_Mo.mat')
load('y_Mo.mat')
 load('Q5.mat')
 p_train=xss(Q(1:103),:)';
t_train=y(Q(1:103),:)';
p_test=xss(Q(104:end),:)';
t_test=y(Q(104:end),:)';

[P_train,in]=mapminmax(p_train);
[T_train,ou]=mapminmax(t_train);




P_test=mapminmax('apply',p_test,in);

tic
net = newff(P_train,T_train,20);

%%
% 2. 设置训练参数
net.trainParam.epochs = 1000;
net.trainParam.goal = 1e-3;
net.trainParam.lr = 0.01;

%%
% 3. 训练网络
net = train(net,P_train,T_train);

%%
% 4. 仿真测试
Tn_test = sim(net,P_test);

%%
% 5. 数据反归一化
T = mapminmax('reverse',Tn_test,ou);
toc
result = [t_test' T'];
%均方误差
E = sqrt(mse(t_test - T))
RMS=std(t_test - T);
N = length(T);
T1=T;
L = length(t_test);  
 R2=(L*sum(T1.*t_test)-sum(T1)*sum(t_test))/sqrt(((L*sum((T1).^2)-(sum(T1))^2)*(L*sum((t_test).^2)-(sum(t_test))^2)))

figure(1)
plot(1:N,t_test,'r-*',1:N,T,'b:o')
grid on
legend('真实值','预测值')
xlabel('样本编号')
ylabel('铜含量')
string = {'测试集铜含量含量预测结果对比(ELM)';['(mse = ' num2str(E) ' R^2 = ' num2str(R2) ')']};
title(string)
