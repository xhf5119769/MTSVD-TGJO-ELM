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

N=20;%%隐含层节点
[R,Q] = size(P_train);
IW = rand(N,R) * 2 - 1;
B = rand(N,1);
BiasMatrix = repmat(B,1,Q);
tempH = IW * P_train + BiasMatrix;
H = 1 ./ (1 + exp(-tempH));
TF='sig';
TYPE=0;

%% ELM算法
LW1 =pinv(H') * T_train';
Tn_test = elmpredict(P_test,IW,B,LW1,TF,TYPE);
T1 = mapminmax('reverse',Tn_test,ou);

E = sqrt(mse(t_test - T1))
RMS=std(t_test - T1);

L = length(t_test);  
 R2=(L*sum(T1.*t_test)-sum(T1)*sum(t_test))/sqrt(((L*sum((T1).^2)-(sum(T1))^2)*(L*sum((t_test).^2)-(sum(t_test))^2)))

figure(1)
plot(1:L,t_test,'r-*',1:L,T,'b:o')
grid on
legend('真实值','预测值')
xlabel('样本编号')
ylabel('铜含量')
string = {'测试集铜含量含量预测结果对比(ELM)';['(mse = ' num2str(E) ' R^2 = ' num2str(R2) ')']};
title(string)

