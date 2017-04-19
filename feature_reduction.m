function features=feature_reduction(sampleindex,sampleindex_a,sampleindex_b)
[feature_train_a_patch,feature_train_b_patch,feature_test_a_patch, feature_test_b_patch]=feature_reduction_patch(sampleindex,sampleindex_a,sampleindex_b);
features{1}.feature_train_a=feature_train_a_patch;
features{1}.feature_train_b=feature_train_b_patch;
features{1}.feature_test_a=feature_test_a_patch;
features{1}.feature_test_b=feature_test_b_patch;

[feature_train_a_latent,feature_train_b_latent,feature_test_a_latent, feature_test_b_latent]=feature_reduction_latent(sampleindex,sampleindex_a,sampleindex_b);
features{2}.feature_train_a=feature_train_a_latent;
features{2}.feature_train_b=feature_train_b_latent;
features{2}.feature_test_a=feature_test_a_latent;
features{2}.feature_test_b=feature_test_b_latent;

[feature_train_a_leg,feature_train_b_leg,feature_test_a_leg, feature_test_b_leg]=feature_reduction_leg(sampleindex,sampleindex_a,sampleindex_b);
features{3}.feature_train_a=feature_train_a_leg;
features{3}.feature_train_b=feature_train_b_leg;
features{3}.feature_test_a=feature_test_a_leg;
features{3}.feature_test_b=feature_test_b_leg;

[feature_train_a_holistic,feature_train_b_holistic,feature_test_a_holistic, feature_test_b_holistic]=feature_reduction_holistic(sampleindex,sampleindex_a,sampleindex_b);
features{4}.feature_train_a=feature_train_a_holistic;
features{4}.feature_train_b=feature_train_b_holistic;
features{4}.feature_test_a=feature_test_a_holistic;
features{4}.feature_test_b=feature_test_b_holistic;


end

function [feature_train_a,feature_train_b,feature_test_a, feature_test_b]=feature_reduction_patch(sampleindex,sampleindex_a,sampleindex_b)
sample_part=load('sample_part');
addpath(genpath('code'));
step=7;
%% partitate the features into training and testing set
for i=1:length(sampleindex)    
           feature_train_1_a(:,(i-1)*step+1:i*step)=[sample_part.sample_part.sample_part_a{sampleindex(i)}(:,:,1)];   
           feature_train_2_a(:,(i-1)*step+1:i*step)=[sample_part.sample_part.sample_part_a{sampleindex(i)}(:,:,2)];
           feature_train_3_a(:,(i-1)*step+1:i*step)=[sample_part.sample_part.sample_part_a{sampleindex(i)}(:,:,3)];
           feature_train_4_a(:,(i-1)*step+1:i*step)=[sample_part.sample_part.sample_part_a{sampleindex(i)}(:,:,4)];
           feature_train_5_a(:,(i-1)*step+1:i*step)=[sample_part.sample_part.sample_part_a{sampleindex(i)}(:,:,5)];
           feature_train_6_a(:,(i-1)*step+1:i*step)=[sample_part.sample_part.sample_part_a{sampleindex(i)}(:,:,6)];
           
           feature_train_1_b(:,(i-1)*step+1:i*step)=[sample_part.sample_part.sample_part_b{sampleindex(i)}(:,:,1)]; 
           feature_train_2_b(:,(i-1)*step+1:i*step)=[sample_part.sample_part.sample_part_b{sampleindex(i)}(:,:,2)];  
           feature_train_3_b(:,(i-1)*step+1:i*step)=[sample_part.sample_part.sample_part_b{sampleindex(i)}(:,:,3)];   
           feature_train_4_b(:,(i-1)*step+1:i*step)=[sample_part.sample_part.sample_part_b{sampleindex(i)}(:,:,4)];  
           feature_train_5_b(:,(i-1)*step+1:i*step)=[sample_part.sample_part.sample_part_b{sampleindex(i)}(:,:,5)]; 
           feature_train_6_b(:,(i-1)*step+1:i*step)=[sample_part.sample_part.sample_part_b{sampleindex(i)}(:,:,6)]; 
end
       
for i=1:length(sampleindex_a)
           feature_test_1_a(:,(i-1)*step*1+1:i*step*1)=[sample_part.sample_part.sample_part_a{sampleindex_a(i)}(:,:,1)];   
           feature_test_2_a(:,(i-1)*step*1+1:i*step*1)=[sample_part.sample_part.sample_part_a{sampleindex_a(i)}(:,:,2)];   
           feature_test_3_a(:,(i-1)*step*1+1:i*step*1)=[sample_part.sample_part.sample_part_a{sampleindex_a(i)}(:,:,3)];   
           feature_test_4_a(:,(i-1)*step*1+1:i*step*1)=[sample_part.sample_part.sample_part_a{sampleindex_a(i)}(:,:,4)];   
           feature_test_5_a(:,(i-1)*step*1+1:i*step*1)=[sample_part.sample_part.sample_part_a{sampleindex_a(i)}(:,:,5)];
           feature_test_6_a(:,(i-1)*step*1+1:i*step*1)=[sample_part.sample_part.sample_part_a{sampleindex_a(i)}(:,:,6)];
end
       
 for i=1:length(sampleindex_b)
           feature_test_1_b(:,(i-1)*step*1+1:i*step*1)=[sample_part.sample_part.sample_part_b{sampleindex_b(i)}(:,:,1)];   
           feature_test_2_b(:,(i-1)*step*1+1:i*step*1)=[sample_part.sample_part.sample_part_b{sampleindex_b(i)}(:,:,2)];   
           feature_test_3_b(:,(i-1)*step*1+1:i*step*1)=[sample_part.sample_part.sample_part_b{sampleindex_b(i)}(:,:,3)];   
           feature_test_4_b(:,(i-1)*step*1+1:i*step*1)=[sample_part.sample_part.sample_part_b{sampleindex_b(i)}(:,:,4)];   
           feature_test_5_b(:,(i-1)*step*1+1:i*step*1)=[sample_part.sample_part.sample_part_b{sampleindex_b(i)}(:,:,5)];
           feature_test_6_b(:,(i-1)*step*1+1:i*step*1)=[sample_part.sample_part.sample_part_b{sampleindex_b(i)}(:,:,6)];
 end
 
 %% perform dimension reduction via PCA
 %% the first stripe
 uxtr=[feature_train_1_a feature_train_1_b];
m=mean(uxtr,2);
number=size(uxtr,2);
cenx=uxtr-repmat(m,1, number);
covx=cov(cenx');
[pcaU value latent]=pcacov(covx);
for i=1:length(latent)
if sum(latent(1:i))>=95
    break;
end
end
dim1=100;
feature_train_a{1}=pcaU(:,1:dim1)'*(feature_train_1_a-repmat(m,1,size(feature_train_1_a,2)));
feature_train_b{1}=pcaU(:,1:dim1)'*(feature_train_1_b-repmat(m,1,size(feature_train_1_b,2)));
feature_test_a{1}=pcaU(:,1:dim1)'*(feature_test_1_a-repmat(m,1,size(feature_test_1_a,2)));
feature_test_b{1}=pcaU(:,1:dim1)'*(feature_test_1_b-repmat(m,1,size(feature_test_1_b,2)));
feature_train_a{1}=normc(feature_train_a{1});
feature_train_b{1}=normc(feature_train_b{1});
feature_test_a{1}=normc(feature_test_a{1});
feature_test_b{1}=normc(feature_test_b{1});

%% the second stripe
uxtr=[feature_train_2_a feature_train_2_b];
m=mean(uxtr,2);
number=size(uxtr,2);
cenx=uxtr-repmat(m,1, number);
covx=cov(cenx');
[pcaU value latent]=pcacov(covx);
dim2=100;

feature_train_a{2}=pcaU(:,1:dim2)'*(feature_train_2_a-repmat(m,1,size(feature_train_2_a,2)));
feature_train_b{2}=pcaU(:,1:dim2)'*(feature_train_2_b-repmat(m,1,size(feature_train_2_b,2)));
feature_test_a{2}=pcaU(:,1:dim2)'*(feature_test_2_a-repmat(m,1,size(feature_test_2_a,2)));
feature_test_b{2}=pcaU(:,1:dim2)'*(feature_test_2_b-repmat(m,1,size(feature_test_2_b,2)));
feature_train_a{2}=normc(feature_train_a{2});
feature_train_b{2}=normc(feature_train_b{2});
feature_test_a{2}=normc(feature_test_a{2});
feature_test_b{2}=normc(feature_test_b{2});

%% the third stripe
uxtr=[feature_train_3_a feature_train_3_b];
m=mean(uxtr,2);
number=size(uxtr,2);
cenx=uxtr-repmat(m,1, number);
covx=cov(cenx');
[pcaU value latent]=pcacov(covx);
dim3=100;

feature_train_a{3}=pcaU(:,1:dim3)'*(feature_train_3_a-repmat(m,1,size(feature_train_3_a,2)));
feature_train_b{3}=pcaU(:,1:dim3)'*(feature_train_3_b-repmat(m,1,size(feature_train_3_b,2)));
feature_test_a{3}=pcaU(:,1:dim3)'*(feature_test_3_a-repmat(m,1,size(feature_test_3_a,2)));
feature_test_b{3}=pcaU(:,1:dim3)'*(feature_test_3_b-repmat(m,1,size(feature_test_3_b,2)));
feature_train_a{3}=normc(feature_train_a{3});
feature_train_b{3}=normc(feature_train_b{3});
feature_test_a{3}=normc(feature_test_a{3});
feature_test_b{3}=normc(feature_test_b{3});


%% the fourth stripe
uxtr=[feature_train_4_a feature_train_4_b];
m=mean(uxtr,2);
number=size(uxtr,2);
cenx=uxtr-repmat(m,1, number);
covx=cov(cenx');
[pcaU value latent]=pcacov(covx);
dim4=100;

feature_train_a{4}=pcaU(:,1:dim4)'*(feature_train_4_a-repmat(m,1,size(feature_train_4_a,2)));
feature_train_b{4}=pcaU(:,1:dim4)'*(feature_train_4_b-repmat(m,1,size(feature_train_4_b,2)));
feature_test_a{4}=pcaU(:,1:dim4)'*(feature_test_4_a-repmat(m,1,size(feature_test_4_a,2)));
feature_test_b{4}=pcaU(:,1:dim4)'*(feature_test_4_b-repmat(m,1,size(feature_test_4_b,2)));
feature_train_a{4}=normc(feature_train_a{4});
feature_train_b{4}=normc(feature_train_b{4});
feature_test_a{4}=normc(feature_test_a{4});
feature_test_b{4}=normc(feature_test_b{4});


%% the fifth stripe
uxtr=[feature_train_5_a feature_train_5_b];
m=mean(uxtr,2);
number=size(uxtr,2);
cenx=uxtr-repmat(m,1, number);
covx=cov(cenx');
[pcaU value latent]=pcacov(covx);
dim5=100;

feature_train_a{5}=pcaU(:,1:dim5)'*(feature_train_5_a-repmat(m,1,size(feature_train_5_a,2)));
feature_train_b{5}=pcaU(:,1:dim5)'*(feature_train_5_b-repmat(m,1,size(feature_train_5_b,2)));
feature_test_a{5}=pcaU(:,1:dim5)'*(feature_test_5_a-repmat(m,1,size(feature_test_5_a,2)));
feature_test_b{5}=pcaU(:,1:dim5)'*(feature_test_5_b-repmat(m,1,size(feature_test_5_b,2)));
feature_train_a{5}=normc(feature_train_a{5});
feature_train_b{5}=normc(feature_train_b{5});
feature_test_a{5}=normc(feature_test_a{5});
feature_test_b{5}=normc(feature_test_b{5});

%% the sixth stripe
uxtr=[feature_train_6_a feature_train_6_b];
m=mean(uxtr,2);
number=size(uxtr,2);
cenx=uxtr-repmat(m,1, number);
covx=cov(cenx');
[pcaU value latent]=pcacov(covx);
dim6=100;

feature_train_a{6}=pcaU(:,1:dim6)'*(feature_train_6_a-repmat(m,1,size(feature_train_6_a,2)));
feature_train_b{6}=pcaU(:,1:dim6)'*(feature_train_6_b-repmat(m,1,size(feature_train_6_b,2)));
feature_test_a{6}=pcaU(:,1:dim6)'*(feature_test_6_a-repmat(m,1,size(feature_test_6_a,2)));
feature_test_b{6}=pcaU(:,1:dim6)'*(feature_test_6_b-repmat(m,1,size(feature_test_6_b,2)));
feature_train_a{6}=normc(feature_train_a{6});
feature_train_b{6}=normc(feature_train_b{6});
feature_test_a{6}=normc(feature_test_a{6});
feature_test_b{6}=normc(feature_test_b{6});
end


function [feature_train_a,feature_train_b,feature_test_a, feature_test_b]=feature_reduction_latent(sampleindex,sampleindex_a,sampleindex_b)
sample_latent=load('sample_latent');
step=5;
for i=1:length(sampleindex)   
           feature_train_1_a(:,(i-1)*step+1:i*step)=[sample_latent.sample_latent.sample_latent_a{sampleindex(i)}(:,:,1)];   
           feature_train_2_a(:,(i-1)*step+1:i*step)=[sample_latent.sample_latent.sample_latent_a{sampleindex(i)}(:,:,2)];
           feature_train_3_a(:,(i-1)*step+1:i*step)=[sample_latent.sample_latent.sample_latent_a{sampleindex(i)}(:,:,3)];
           feature_train_4_a(:,(i-1)*step+1:i*step)=[sample_latent.sample_latent.sample_latent_a{sampleindex(i)}(:,:,4)];
           feature_train_5_a(:,(i-1)*step+1:i*step)=[sample_latent.sample_latent.sample_latent_a{sampleindex(i)}(:,:,5)];
           
           feature_train_1_b(:,(i-1)*step+1:i*step)=[sample_latent.sample_latent.sample_latent_b{sampleindex(i)}(:,:,1)]; 
           feature_train_2_b(:,(i-1)*step+1:i*step)=[sample_latent.sample_latent.sample_latent_b{sampleindex(i)}(:,:,2)];  
           feature_train_3_b(:,(i-1)*step+1:i*step)=[sample_latent.sample_latent.sample_latent_b{sampleindex(i)}(:,:,3)];   
           feature_train_4_b(:,(i-1)*step+1:i*step)=[sample_latent.sample_latent.sample_latent_b{sampleindex(i)}(:,:,4)];  
           feature_train_5_b(:,(i-1)*step+1:i*step)=[sample_latent.sample_latent.sample_latent_b{sampleindex(i)}(:,:,5)]; 
end
for i=1:length(sampleindex_a)
           feature_test_1_a(:,(i-1)*step*1+1:i*step*1)=[sample_latent.sample_latent.sample_latent_a{sampleindex_a(i)}(:,:,1)];   
           feature_test_2_a(:,(i-1)*step*1+1:i*step*1)=[sample_latent.sample_latent.sample_latent_a{sampleindex_a(i)}(:,:,2)];   
           feature_test_3_a(:,(i-1)*step*1+1:i*step*1)=[sample_latent.sample_latent.sample_latent_a{sampleindex_a(i)}(:,:,3)];   
           feature_test_4_a(:,(i-1)*step*1+1:i*step*1)=[sample_latent.sample_latent.sample_latent_a{sampleindex_a(i)}(:,:,4)];   
           feature_test_5_a(:,(i-1)*step*1+1:i*step*1)=[sample_latent.sample_latent.sample_latent_a{sampleindex_a(i)}(:,:,5)];
end
 for i=1:length(sampleindex_b)
           feature_test_1_b(:,(i-1)*step*1+1:i*step*1)=[sample_latent.sample_latent.sample_latent_b{sampleindex_b(i)}(:,:,1)];   
           feature_test_2_b(:,(i-1)*step*1+1:i*step*1)=[sample_latent.sample_latent.sample_latent_b{sampleindex_b(i)}(:,:,2)];   
           feature_test_3_b(:,(i-1)*step*1+1:i*step*1)=[sample_latent.sample_latent.sample_latent_b{sampleindex_b(i)}(:,:,3)];   
           feature_test_4_b(:,(i-1)*step*1+1:i*step*1)=[sample_latent.sample_latent.sample_latent_b{sampleindex_b(i)}(:,:,4)];   
           feature_test_5_b(:,(i-1)*step*1+1:i*step*1)=[sample_latent.sample_latent.sample_latent_b{sampleindex_b(i)}(:,:,5)];
 end
 
 uxtr=[feature_train_1_a feature_train_1_b];
m=mean(uxtr,2);
number=size(uxtr,2);
cenx=uxtr-repmat(m,1, number);
covx=cov(cenx');
[pcaU value latent]=pcacov(covx);
dim1=100;
feature_train_a{1}=pcaU(:,1:dim1)'*(feature_train_1_a-repmat(m,1,size(feature_train_1_a,2)));
feature_train_b{1}=pcaU(:,1:dim1)'*(feature_train_1_b-repmat(m,1,size(feature_train_1_b,2)));
feature_test_a{1}=pcaU(:,1:dim1)'*(feature_test_1_a-repmat(m,1,size(feature_test_1_a,2)));
feature_test_b{1}=pcaU(:,1:dim1)'*(feature_test_1_b-repmat(m,1,size(feature_test_1_b,2)));
feature_train_a{1}=normc(feature_train_a{1});
feature_train_b{1}=normc(feature_train_b{1});
feature_test_a{1}=normc(feature_test_a{1});
feature_test_b{1}=normc(feature_test_b{1});

uxtr=[feature_train_2_a feature_train_2_b];
m=mean(uxtr,2);
number=size(uxtr,2);
cenx=uxtr-repmat(m,1, number);
covx=cov(cenx');
[pcaU value latent]=pcacov(covx);
dim2=100;

feature_train_a{2}=pcaU(:,1:dim2)'*(feature_train_2_a-repmat(m,1,size(feature_train_2_a,2)));
feature_train_b{2}=pcaU(:,1:dim2)'*(feature_train_2_b-repmat(m,1,size(feature_train_2_b,2)));
feature_test_a{2}=pcaU(:,1:dim2)'*(feature_test_2_a-repmat(m,1,size(feature_test_2_a,2)));
feature_test_b{2}=pcaU(:,1:dim2)'*(feature_test_2_b-repmat(m,1,size(feature_test_2_b,2)));
feature_train_a{2}=normc(feature_train_a{2});
feature_train_b{2}=normc(feature_train_b{2});
feature_test_a{2}=normc(feature_test_a{2});
feature_test_b{2}=normc(feature_test_b{2});


uxtr=[feature_train_3_a feature_train_3_b];
m=mean(uxtr,2);
number=size(uxtr,2);
cenx=uxtr-repmat(m,1, number);
covx=cov(cenx');
[pcaU value latent]=pcacov(covx);
dim3=100;

feature_train_a{3}=pcaU(:,1:dim3)'*(feature_train_3_a-repmat(m,1,size(feature_train_3_a,2)));
feature_train_b{3}=pcaU(:,1:dim3)'*(feature_train_3_b-repmat(m,1,size(feature_train_3_b,2)));
feature_test_a{3}=pcaU(:,1:dim3)'*(feature_test_3_a-repmat(m,1,size(feature_test_3_a,2)));
feature_test_b{3}=pcaU(:,1:dim3)'*(feature_test_3_b-repmat(m,1,size(feature_test_3_b,2)));
feature_train_a{3}=normc(feature_train_a{3});
feature_train_b{3}=normc(feature_train_b{3});
feature_test_a{3}=normc(feature_test_a{3});
feature_test_b{3}=normc(feature_test_b{3});

uxtr=[feature_train_4_a feature_train_4_b];
m=mean(uxtr,2);
number=size(uxtr,2);
cenx=uxtr-repmat(m,1, number);
covx=cov(cenx');
[pcaU value latent]=pcacov(covx);
dim4=100;

feature_train_a{4}=pcaU(:,1:dim4)'*(feature_train_4_a-repmat(m,1,size(feature_train_4_a,2)));
feature_train_b{4}=pcaU(:,1:dim4)'*(feature_train_4_b-repmat(m,1,size(feature_train_4_b,2)));
feature_test_a{4}=pcaU(:,1:dim4)'*(feature_test_4_a-repmat(m,1,size(feature_test_4_a,2)));
feature_test_b{4}=pcaU(:,1:dim4)'*(feature_test_4_b-repmat(m,1,size(feature_test_4_b,2)));
feature_train_a{4}=normc(feature_train_a{4});
feature_train_b{4}=normc(feature_train_b{4});
feature_test_a{4}=normc(feature_test_a{4});
feature_test_b{4}=normc(feature_test_b{4});

uxtr=[feature_train_5_a feature_train_5_b];
m=mean(uxtr,2);
number=size(uxtr,2);
cenx=uxtr-repmat(m,1, number);
covx=cov(cenx');
[pcaU value latent]=pcacov(covx);
dim5=100;

feature_train_a{5}=pcaU(:,1:dim5)'*(feature_train_5_a-repmat(m,1,size(feature_train_5_a,2)));
feature_train_b{5}=pcaU(:,1:dim5)'*(feature_train_5_b-repmat(m,1,size(feature_train_5_b,2)));
feature_test_a{5}=pcaU(:,1:dim5)'*(feature_test_5_a-repmat(m,1,size(feature_test_5_a,2)));
feature_test_b{5}=pcaU(:,1:dim5)'*(feature_test_5_b-repmat(m,1,size(feature_test_5_b,2)));
feature_train_a{5}=normc(feature_train_a{5});
feature_train_b{5}=normc(feature_train_b{5});
feature_test_a{5}=normc(feature_test_a{5});
feature_test_b{5}=normc(feature_test_b{5});
end

function [feature_train_a,feature_train_b,feature_test_a, feature_test_b]=feature_reduction_leg(sampleindex,sampleindex_a,sampleindex_b)
 data=load('sample_leg_latent.mat');
 step=9;
 
 for i=1:length(sampleindex)
           feature_train_1_a(:,(i-1)*step+1:i*step)=[data.sample_leg_latent.sample_leg_latent_a{sampleindex(i)}.feature_left];   
           feature_train_2_a(:,(i-1)*step+1:i*step)=[data.sample_leg_latent.sample_leg_latent_a{sampleindex(i)}.feature_right];
           feature_train_1_b(:,(i-1)*step+1:i*step)=[data.sample_leg_latent.sample_leg_latent_b{sampleindex(i)}.feature_left]; 
           feature_train_2_b(:,(i-1)*step+1:i*step)=[data.sample_leg_latent.sample_leg_latent_b{sampleindex(i)}.feature_right];  
 end
 
 for i=1:length(sampleindex_a)
           feature_test_1_a(:,(i-1)*step*1+1:i*step*1)=[data.sample_leg_latent.sample_leg_latent_a{sampleindex_a(i)}.feature_left];   
           feature_test_2_a(:,(i-1)*step*1+1:i*step*1)=[data.sample_leg_latent.sample_leg_latent_a{sampleindex_a(i)}.feature_right];   
end
       
for i=1:length(sampleindex_b)
           feature_test_1_b(:,(i-1)*step*1+1:i*step*1)=[data.sample_leg_latent.sample_leg_latent_b{sampleindex_b(i)}.feature_left];   
           feature_test_2_b(:,(i-1)*step*1+1:i*step*1)=[data.sample_leg_latent.sample_leg_latent_b{sampleindex_b(i)}.feature_right];   
end

uxtr=[feature_train_1_a feature_train_1_b];
m=mean(uxtr,2);
number=size(uxtr,2);
cenx=uxtr-repmat(m,1, number);
covx=cov(cenx');
[pcaU value latent]=pcacov(covx);
dim1=100;
feature_train_a{1}=pcaU(:,1:dim1)'*(feature_train_1_a-repmat(m,1,size(feature_train_1_a,2)));
feature_train_b{1}=pcaU(:,1:dim1)'*(feature_train_1_b-repmat(m,1,size(feature_train_1_b,2)));
feature_test_a{1}=pcaU(:,1:dim1)'*(feature_test_1_a-repmat(m,1,size(feature_test_1_a,2)));
feature_test_b{1}=pcaU(:,1:dim1)'*(feature_test_1_b-repmat(m,1,size(feature_test_1_b,2)));
feature_train_a{1}=normc(feature_train_a{1});
feature_train_b{1}=normc(feature_train_b{1});
feature_test_a{1}=normc(feature_test_a{1});
feature_test_b{1}=normc(feature_test_b{1});

uxtr=[feature_train_2_a feature_train_2_b];
m=mean(uxtr,2);
number=size(uxtr,2);
cenx=uxtr-repmat(m,1, number);
covx=cov(cenx');
[pcaU value latent]=pcacov(covx);
dim2=100;

feature_train_a{2}=pcaU(:,1:dim2)'*(feature_train_2_a-repmat(m,1,size(feature_train_2_a,2)));
feature_train_b{2}=pcaU(:,1:dim2)'*(feature_train_2_b-repmat(m,1,size(feature_train_2_b,2)));
feature_test_a{2}=pcaU(:,1:dim2)'*(feature_test_2_a-repmat(m,1,size(feature_test_2_a,2)));
feature_test_b{2}=pcaU(:,1:dim2)'*(feature_test_2_b-repmat(m,1,size(feature_test_2_b,2)));
feature_train_a{2}=normc(feature_train_a{2});
feature_train_b{2}=normc(feature_train_b{2});
feature_test_a{2}=normc(feature_test_a{2});
feature_test_b{2}=normc(feature_test_b{2});
end

function [feature_train_a,feature_train_b,feature_test_a, feature_test_b]=feature_reduction_holistic(sampleindex,sampleindex_a,sampleindex_b)

step=1;
data=load('FeatureSetC');
range=1; length1=length(sampleindex);
for i=1:length(sampleindex)         
           feature_train_1_a(:,(i-1)*step+1:i*step)=[data.FeatureSetC.feature_a(:,sampleindex(i))];  
           feature_train_1_b(:,(i-1)*step+1:i*step)=[data.FeatureSetC.feature_b(:,sampleindex(i))]; 
end
for i=1:length(sampleindex_a)
           feature_test_1_a(:,(i-1)*step+1:i*step)=[data.FeatureSetC.feature_a(:,sampleindex_a(i))];   
 end
 for i=1:length(sampleindex_b)
           feature_test_1_b(:,(i-1)*step+1:i*step)=[data.FeatureSetC.feature_b(:,sampleindex_b(i))];   
 end 
 
 X_a_train_holistic_id= [ floor(((1:length1*range)-0.1)/range)+1 ];
X_b_train_holistic_id= [ floor(((1:length1*range)-0.1)/range)+1 ];
[W, M] = XQDA(feature_train_1_b',  feature_train_1_a', X_b_train_holistic_id', X_a_train_holistic_id');
feature_train_b=normc((feature_train_1_b'*W)');
feature_train_a=normc((feature_train_1_a'*W)');
feature_test_a=normc((feature_test_1_a'*W)');
feature_test_b=normc((feature_test_1_b'*W)');
 
dim1=size(feature_train_a,1);
X_a_train1=feature_train_a; X_b_train1=feature_train_b;
end