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


[P_train,in]=mapminmax(p_train);
[T_train,ou]=mapminmax(t_train);

P_test=mapminmax('apply',p_test,in);
tic
net = newrbe(P_train,T_train,1100);

%%
% 2. 仿真测试
Tn_test = sim(net,P_test);
%反归一化
T = mapminmax('reverse',Tn_test,ou);
toc
%误差
result = [t_test' T'];
%均方误差
E = sqrt(mse(t_test - T))
RMS=std(t_test - T);
N = length(T);
T1=T;
L = length(t_test);  
 R2=(L*sum(T1.*t_test)-sum(T1)*sum(t_test))/sqrt(((L*sum((T1).^2)-(sum(T1))^2)*(L*sum((t_test).^2)-(sum(t_test))^2)))


