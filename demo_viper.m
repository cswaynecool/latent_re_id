clear; clc;

%% we extract features for the holistic and patch models
imgname=['./VIPeR_m/'];
imgname_a=imgname;
imgname_b=imgname;
img_list=dir([imgname,'*.png']);
index=1:632;
img_list_a=img_list(2*index-1);
img_list_b=img_list(2*index);
par.nparts=6;

Feature_Extract=[1 1 1 1];
if Feature_Extract(1)==1
  
        for i=1:length(img_list_a)
            fprintf('computing dense feature for %d-th image ...\n', i); 
            img=imread([imgname_a,img_list_a(i).name]);
             options.numScales=3;
             f=LOMO(img,options);
             feature_a(:,i)=f;
        end
        FeatureSetC.feature_a=feature_a;
        
        for i=1:length(img_list_b)
            fprintf('computing dense feature for %d-th image ...\n', i); 
            img=imread([imgname_b,img_list_b(i).name]);
             options.numScales=3;
             f=LOMO(img,options);
             feature_b(:,i)=f;
        end
        FeatureSetC.feature_b=feature_b;
        save FeatureSetC 'FeatureSetC' -v7.3;
end
%%  extract features for the horizonal patches
if Feature_Extract(2)==1
        for i=1:length(img_list_a)
            fprintf('computing dense feature for %d-th image ...\n', i); 
                img=imread([imgname_a,img_list_a(i).name]);
                f=part_feature_lomo(img);
                sample_part_a{i}=f;
        end
        for i=1:length(img_list_b)
            fprintf('computing dense feature for %d-th image ...\n', i); 
                img=imread([imgname_b,img_list_b(i).name]);
                f=part_feature_lomo(img);
                sample_part_b{i}=f;
        end
        sample_part.sample_part_a=sample_part_a; sample_part.sample_part_b=sample_part_b;
        save sample_part 'sample_part' -v7.3;
end

%% extract features for the vertival patches
if Feature_Extract(3)==1
        for i=1:length(img_list_a)
            fprintf('computing upper Latent feature for %d-th image ...\n', i);         
            img=imread([imgname_a,img_list_a(i).name]);
            sample_latent_a{i}=feature_upper_latent_lomo(img);
        end
         for i=1:length(img_list_b)
            fprintf('computing upper Latent feature for %d-th image ...\n', i);
            img=imread([imgname_b,img_list_b(i).name]);
            sample_latent_b{i}=feature_upper_latent_lomo(img);
         end
        sample_latent.sample_latent_a=sample_latent_a;
        sample_latent.sample_latent_b=sample_latent_b;
        save sample_latent 'sample_latent' -v7.3;
end

%% extract features for the leg postures
if Feature_Extract(4)==1
        for i=1:length(img_list_a)
            fprintf('computing upper Latent feature for %d-th image ...\n', i);
            img=imread([imgname_a,img_list_a(i).name]);
            sample_leg_latent_a{i}=feature_leg_latent(img);
        end
        for i=1:length(img_list_b)
            fprintf('computing upper Latent feature for %d-th image ...\n', i);
            img=imread([imgname_b,img_list_b(i).name]);
            sample_leg_latent_b{i}=feature_leg_latent(img);
        end
        sample_leg_latent.sample_leg_latent_a=sample_leg_latent_a;
        sample_leg_latent.sample_leg_latent_b=sample_leg_latent_b;
        save sample_leg_latent 'sample_leg_latent';
end

%% partitate the features into training and testing set
for total_loop=1:10
rand('state',total_loop);
a=rand(1,632); [a b]=sort(a);
sampleindex=b(1:316);
sampleindex_a=b(317:end);
sampleindex_b=b(317:end);

fprintf('perform feature reduction...\n');
features=feature_reduction(sampleindex,sampleindex_a,sampleindex_b);

fprintf('training and testing the model...\n');
DistanceMatrix=train_test(sampleindex,features,total_loop);


CountOfGalleryTargetSamples=ones(1,316);
CountOfTestingTargetSamples=ones(1,316);
TestSampleAmount=sum(CountOfTestingTargetSamples);
[ExtentionArrayWithClassLabel1] = StandardExtendClassLableArray(CountOfGalleryTargetSamples);
[SortedDistanceMatrix,Index] = sort(DistanceMatrix,'ascend');
[a1 a2]=size(Index);
Index=reshape(ExtentionArrayWithClassLabel1(Index(:)),a1,a2);
[ExtentionArrayWithClassLabel] = StandardExtendClassLableArray(CountOfTestingTargetSamples);
if size(ExtentionArrayWithClassLabel,1) > size(ExtentionArrayWithClassLabel,2)
    ExtentionArrayWithClassLabel = ExtentionArrayWithClassLabel';
end

for rank = 1 : 20
    Temp = Index(1 : rank,:) == ones(rank,1) * ExtentionArrayWithClassLabel;    
    Rank1cAccRate(rank) = ...
        sum(sum(Temp==1,1)) / TestSampleAmount;
    disp(['MatchingRate for Rank ' num2str(rank) ' is ' num2str(Rank1cAccRate(rank)*100) ' %']);
end
cmc(total_loop,:)=Rank1cAccRate;
end