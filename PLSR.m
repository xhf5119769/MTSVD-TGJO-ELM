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
b=regress(T_train',P_train');
T1=[P_test']*b;
TT=T1';
T = mapminmax('reverse',TT,ou);
E = sqrt(mse(t_test - T))
N = length(T);
T1=T;
L = length(t_test);  
 R2=(L*sum(T1.*t_test)-sum(T1)*sum(t_test))/sqrt(((L*sum((T1).^2)-(sum(T1))^2)*(L*sum((t_test).^2)-(sum(t_test))^2)))