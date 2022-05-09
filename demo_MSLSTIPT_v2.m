%% Stable multi-subspace learning and spatial-temporal tensor model for infrared small target detection
%% V2 version of MSLSTIPT, which uses whole frame of infrared sequences instead of sliding windows
clc;
clear;
close all;

addpath(genpath('proximal_operator'));
addpath(genpath('tSVD'));
addpath(genpath('WSNM_problem'));

% setup parameters
C=5;
p=0.8;
patch_frames=6;
H=4;

%% input data
% strDir='data\';
strDir='I:\²©Ê¿Ñ§Ï°\Yangcode\µÚÆßÆª\new_data\Sequence5\';
%% processing
tic
k=12;
for i=1:k
    picname=[strDir  num2str(i),'.bmp'];
    I=imread(picname);
    [~, ~, ch]=size(I);
    if ch==3
        I=rgb2gray(I);
    end
    D(:,:,i)=I;
end
      tenD=double(D);
        size_D=size(tenD);
        [n1,n2,n3]=size(tenD);
        n_1=max(n1,n2);%n(1)
        n_2=min(n1,n2);%n(2)
        patch_num=n3/patch_frames;
for l=1:patch_num
    temp=zeros(n1,n2,patch_frames);
    for i=1:patch_frames
        temp(:,:,i)=tenD(:,:,patch_frames*(l-1)+i);
    end  
    %% test R-TPCA
    temp=reshape(temp,n1,patch_frames,n2);
    [a,b,c]=size(temp);
    opts.lambda = H/sqrt(max(a,b)*c);
    opts.mu = 1e-4;
    opts.tol = 1e-8;
    opts.rho = 1.2;
    opts.max_iter = 800;
    opts.DEBUG = 0;    
    [ L,E,rank] = dictionary_learning( temp, opts);  
    %% approximate L, since sometimes R-TPCA cannot produce a good dictionary
    tho=50;
    Debug = 0;
    [ L_hat,trank,U,V,S ] = prox_low_rank(L,tho);
    if Debug
        fprintf('\n\n ||L_hat-L||=%.3e,  rank=%d\n\n',tnorm(L_hat-L,'fro'),trank);
    end
    LL=tprod(U,S);%%tinny-tsvd
    %% MSLSTIPT
    max_iter=200;
    TT=C*sqrt(n1*n2);
    [Z,tenE,Z_rank,err_va ] = MSLSTIPT(temp,LL,TT,p,max_iter,Debug);
    tenL=tprod(LL,Z);
    tenE=reshape(tenE,n1,n2,patch_frames);
    tenL=reshape(tenL,n1,n2,patch_frames);
%% Reconstruct
 for i=1:patch_frames
      tarImg=tenE(:,:,i);
      backImg=tenL(:,:,i);
      a=uint8(tarImg);        
      figure;imshow(a, []);
end 
end
toc