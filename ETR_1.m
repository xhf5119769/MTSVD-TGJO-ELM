clc
clear
load('Mo.mat')
 load('Q1.mat')
 p=Data(2:end,:);
 t=Data(1,:);
 
p_train=p(:,Q(1:103));
t_train=t(:,Q(1:103));
p_test=p(:,Q(104:end));
t_test=t(:,Q(104:end));

[P_train,in]=mapminmax(p_train);
[T_train,ou]=mapminmax(t_train);
P_test=mapminmax('apply',p_test,in);

M    =  100; % Number of Extra_Trees in the ensemble
k    =  2;  % Number of attributes selected to perform the random splits 
            % 1 <k <= total number of attributes 
nmin =  1;  % Minimum number of points for each leaf

[ensemble,Tn_train] = buildAnEnsemble(M,k,nmin,[P_train',T_train'],0);

% Run the ensemble on a validation dataset
TT = predictWithAnEnsemble(ensemble,[P_test',t_test'],0);
T = mapminmax('reverse',TT',ou);
result = [t_test' T'];
%¾ù·½Îó²î
E = sqrt(mse(t_test - T))

N = length(T);
T1=T;
L = length(t_test);  
 R2=(L*sum(T1.*t_test)-sum(T1)*sum(t_test))/sqrt(((L*sum((T1).^2)-(sum(T1))^2)*(L*sum((t_test).^2)-(sum(t_test))^2)))
