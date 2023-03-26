clc
clear
load('data_Mo.mat')
load('y_Mo.mat')

Q=randperm(size(xss,1));
p_train=xss(Q(1:103),:)';
t_train=y(Q(1:103),:)';
p_test=xss(Q(104:end),:)';
t_test=y(Q(104:end),:)';
%% 数据归一化
[P_train,in]=mapminmax(p_train);
[T_train,ou]=mapminmax(t_train);
P_test=mapminmax('apply',p_test,in);
[inputnum,samplenum]=size(P_train); %输入特征数量以及样本数量
hiddennum=20;  %隐含层节点数量
num=inputnum*hiddennum+hiddennum;  %总数量

%% SSA参数选择
maxgen=30;   % 迭代次数  
sizepop=30;   %种群规模
popmax=5;
popmin=-5;
p=1; %T分布值
%% 计算适应度函数

for i=1:sizepop
    pop(i,:)=2*rand(1,num)-1;%随机生成参数
    IW=reshape(pop(i,(1:inputnum*hiddennum)),hiddennum,inputnum);
    B=(pop(i,inputnum*hiddennum+1:end))';
    BiasMatrix=repmat(B,1,samplenum);
    tempH = IW * P_train + BiasMatrix;
     H = 1 ./ (1 + exp(-tempH));
     LW=pinv(H') * T_train';
     TF='sig';
     TYPE=0;
     t = elmpredict(P_test,IW,B,LW,TF,TYPE);
     T = mapminmax('reverse',t,ou);
     E=sqrt(mse(t_test - T));%适应度值
     fitness(i)=E;
end
[ ans, sortIndex ]=sort( fitness );% 将适应度值从小向大排列
male_index=sortIndex(1);% 确定雄性豺狼位置
female_index=sortIndex(2);% 确定雌性豺狼位置
male_pop=pop(male_index,:);% 确定雄性豺狼位置
female_pop=pop(female_index,:);% 确定雌性豺狼位置
fitness_male=fitness(male_index);% 将适应度值最优的为雄性豺狼位置
fitness_female=fitness(female_index);% 将适应度值次优的为雄性豺狼位置
pFit = fitness;    
ffMin=fitness_male;
%% SSA迭代寻优
a=0;

for i=1:size(pop,1)
   for j=1:size(pop,2)
          E1=1.5*(1-(a/maxgen));
   RL=0.05*levy(maxgen,num,1.5);
    r1=rand();
     E0=2*r1-1;
      E=E1*E0; 
     if abs(E)<1
          D_male_jackal=abs((RL(i,j)*male_pop(j)-pop(i,j))); 
                Male_Positions(i,j)=male_pop(j)-E*D_male_jackal;
                D_female_jackal=abs((RL(i,j)*female_pop(j)-pop(i,j))); 
                Female_Positions(i,j)=female_pop(j)-E*D_female_jackal;
           else
                %% EXPLORATION
               D_male_jackal=abs( (male_pop(j)- RL(i,j)*pop(i,j)));
                Male_Positions(i,j)=male_pop(j)-E*D_male_jackal;
              D_female_jackal=abs( (female_pop(j)- RL(i,j)*pop(i,j)));
                Female_Positions(i,j)=female_pop(j)-E*D_female_jackal;
     end
        pop(i,j)=(Male_Positions(i,j)+Female_Positions(i,j))/2;
             IW=reshape(pop(i,(1:inputnum*hiddennum)),hiddennum,inputnum);
    B=(pop(i,inputnum*hiddennum+1:end))';
    BiasMatrix=repmat(B,1,samplenum);
    tempH = IW * P_train + BiasMatrix;
     H = 1 ./ (1 + exp(-tempH));
     LW=pinv(H') * T_train';
     TF='sig';
     TYPE=0;
     t = elmpredict(P_test,IW,B,LW,TF,TYPE);
     T = mapminmax('reverse',t,ou);
     E=sqrt(mse(t_test - T));%适应度值
     fitness(i)=E;
     
[ ans, sortIndex ]=sort( fitness );% 将适应度值从小向大排列
male_index=sortIndex(1);% 确定雄性豺狼位置
female_index=sortIndex(2);% 确定雌性豺狼位置
male_pop=pop(male_index,:);% 确定雄性豺狼位置
female_pop=pop(female_index,:);% 确定雌性豺狼位置
fitness_male=fitness(male_index);% 将适应度值最优的为雄性豺狼位置
fitness_female=fitness(female_index);% 将适应度值次优的为雄性豺狼位置
   end

    for ii = 1 : sizepop 
        if ( fitness( ii ) < pFit( ii ) )
            pFit( ii ) = fitness( ii );
        end
        
        if( pFit(ii) < fitness_male )
           fitness_male= pFit(ii);
            male_pop =pop(ii, : );
        end
    end
           
       a=a+1;
       yy(a)=fitness_male;    
              % ffMin=fitness_male;
           fit=fitness;
    ppFit=pFit;
          %自适应t分布变异
    for jj = 1:sizepop
        if rand < p
           Temp(jj,:) = pop(jj,:) + pop(jj,:)*trnd(a); %基于迭代次数的t分布变异
           IW=reshape(Temp(jj,(1:inputnum*hiddennum)),hiddennum,inputnum);
           B=(Temp(jj,inputnum*hiddennum+1:end))';
         BiasMatrix=repmat(B,1,samplenum);
         tempH = IW * P_train + BiasMatrix;  
         H = 1 ./ (1 + exp(-tempH));   
         LW=pinv(H') * T_train';   
         TF='sig';
         TYPE=0;
         t = elmpredict(P_test,IW,B,LW,TF,TYPE);
         T = mapminmax('reverse',t,ou);
         E=sqrt(mse(t_test - T));
         fitvalue=E;  
               if(fitvalue <fit(jj))   %如果小于适应度值
               fit(jj) = fitvalue;
           end
        else
        Temp(jj,:) = pop(jj,:);
          IW=reshape(Temp(jj,(1:inputnum*hiddennum)),hiddennum,inputnum);
           B=(Temp(jj,inputnum*hiddennum+1:end))';
         BiasMatrix=repmat(B,1,samplenum);
         tempH = IW * P_train + BiasMatrix;  
         H = 1 ./ (1 + exp(-tempH));   
         LW=pinv(H') * T_train';   
         TF='sig';
         TYPE=0;
         t = elmpredict(P_test,IW,B,LW,TF,TYPE);
         T = mapminmax('reverse',t,ou);
         E=sqrt(mse(t_test - T));
         fitvalue=E;  
           if(fitvalue <fit(jj))   %如果小于适应度值
               fit(jj) = fitvalue;
           end
        end    
    end
      
    for ia = 1 : sizepop 
        if ( fit( ia ) < ppFit( ia ) )
            ppFit( ia ) = fit( ia );
        end
        
        if( ppFit( ia ) < ffMin )
           ffMin= ppFit( ia );
           bestXX =Temp( ia, : );
        end
    end
 
    yyy(a)=ffMin;    

    
end
figure(1)   
plot(yy)
hold on
plot(yyy)
legend('GJO-ELM','TGJO-ELM')

IW=reshape(bestXX((1:inputnum*hiddennum)),hiddennum,inputnum);
 B=(bestXX(:,inputnum*hiddennum+1:end))';
BiasMatrix=repmat(B,1,samplenum);
tempH = IW * P_train + BiasMatrix;  
H = 1 ./ (1 + exp(-tempH));   
         
LW=pinv(H') * T_train';   
TF='sig';
TYPE=0;
 t = elmpredict(P_test,IW,B,LW,TF,TYPE);
 T = mapminmax('reverse',t,ou);
E=sqrt(mse(t_test - T));
N=length(T);
R2=((N*sum(T.*t_test)-sum(T)*sum(t_test))/sqrt(((N*sum((T).^2)-(sum(T))^2)*(N*sum((t_test).^2)-(sum(t_test))^2))));
NN=hiddennum;
 [U,S,V]=svd(H');
 s=diag(pinv(S));
 LW5=zeros(NN,1);
 R5=[];
 R6=[];
R7=[];
LW=zeros(NN,1);
         for i=1:NN
    LWi=V(:,i)*s(i)*U(:,i)'*T_train';
    LW=LWi+LW;
   Tn_test5 = elmpredict(P_test,IW,B,LW,TF,TYPE);
   T5 = mapminmax('reverse',Tn_test5,ou);
   E5 = mse(t_test - T5);
   R=((N*sum(T5.*t_test)-sum(T5)*sum(t_test))/sqrt(((N*sum((T5).^2)-(sum(T5))^2)*(N*sum((t_test).^2)-(sum(t_test))^2))));   
  R5=[R5;E5];
  R6=[R6;T5];
  R7=[R7;R];
         end

[c,d]=max(R7)
E5=R5(d,:);
T5=R6(d,:);
E5=sqrt(R5(d,:)); 

%% 当NN特别大的时候，可以执行以下程序（MTSVD）

% while  d<NN
 %    q1=find(diag(S)<(S(d,d)/2));
% dd=min(q1);
% for q=d:(dd-1)
  %      S(q,q)=S(d,d);
% end
% for q1=dd:NN
 %       S(q1,q1)=0;
% end
% break
% end
% LWW=pinv(U*S*V')*T_train';
% Tn_test5 = elmpredict(P_test,IW,B,LWW,TF,TYPE);
% T5 = mapminmax('reverse',Tn_test5,ou);
% E5 = mse(t_test - T5);
% R=((N*sum(T5.*t_test)-sum(T5)*sum(t_test))/sqrt(((N*sum((T5).^2)-(sum(T5))^2)*(N*sum((t_test).^2)-(sum(t_test))^2)))) 
    
    
     
         
         figure (2)
         plot(1:N,t_test,'r-*',1:N,T5,'b:o')
         grid on
         legend('真实值','预测值')
         xlabel('样本编号')
         ylabel('铜含量')
         string = {'测试集铜含量含量预测结果对比(ELM)';['(mse = ' num2str(E5) ' R^2 = ' num2str(c) ')']};
         title(string)

         
         