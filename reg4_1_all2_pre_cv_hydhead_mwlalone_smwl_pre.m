close all;clear all;clc;
clear all;close all;clc
[~,~,well]=xlsread('C:\Users\WEM\OneDrive\phd_papers\Acta_geophysica\JOH_p2\dtw_ts_co_tradp_full.xls','hyd_head1');
% [X,map,R]=geotiffread('D:\mohanasundaram\paper6_maps\gis\images\export\mask.tif');
% [X1,map1,R1]=geotiffread('D:\mohanasundaram\paper6_maps\gis\images\export\dem_new_f1_1000.tif');
% % [X2,map2,R2]=geotiffread('D:\mohanasundaram\paper6_maps\gis\images\export\soil.tif');
% % X=double(X);
% [X,map,R]=geotiffread('D:\mohanasundaram\paper5_maps\manuscript_submission_p5\paper5_maps\gis\gis_layers\Extract_bound1.tif');
[X1,map1,R1]=geotiffread('C:\Users\WEM\OneDrive\phd_papers\Acta_geophysica\JOH_p2\gis\maps\dem_90_1.tif');
[X,map,R]=geotiffread('C:\Users\WEM\OneDrive\phd_papers\Acta_geophysica\JOH_p2\gis\maps\bound_90_4.tif');
[~,~,ints]=xlsread('C:\Users\WEM\OneDrive\phd_papers\Acta_geophysica\JOH_p2\gis\trans_xy2.xlsx','trans_xy2');

% X1=X1+10;
% X2=double(X2);
% [~,~,raindata]=xlsread('D:\mohanasundaram\paper6_maps\data\rain_data1.xlsx','rain_data1');
% % [~,~,wldata]=xlsread('D:\mohanasundaram\paper6_maps\data\wl_data.xlsx','wl_data');
% [~,~,coeff]=xlsread('D:\mohanasundaram\paper6_maps\data\coeff.xlsx','coeff');
% [~,~,ref]=xlsread('D:\mohanasundaram\paper6_maps\data\wells_adyar1_ref.xlsx','wells_adyar1_ref');
% [X,map,R]=geotiffread('D:\mohanasundaram\paper6_maps\gis\images\bound_ras.tif');
% [X1,map1,R1]=geotiffread('D:\mohanasundaram\paper6_maps\gis\images\dem_4.tif');

% X=X(3:end-3,3:end-3);
% X1=X1(3:end-3,3:end-3);
x=well(2:end,3);
x=cell2mat(x);
y=well(2:end,4);
y=cell2mat(y);
% dem=well(2:end,7);
% dem=cell2mat(dem)+10;
wl=well(2:end,9:end);
wl=cell2mat(wl);
wl(wl<0)=0.05;

%line_coord
x1=ints(2:end,2);
x1=cell2mat(x1);
y1=ints(2:end,3);
y1=cell2mat(y1);
D1=ints(2:end,7);
D1=cell2mat(D1);

dem_wells=well(2:end,7);
dem_wells=cell2mat(dem_wells);

%create meshgrid
minx=R1(1,1);
maxx=R1(2,1);
miny=R1(1,2);
maxy=R1(2,2);
dx=90;dy=-90;
[xii,yii]=meshgrid(minx:dx:maxx-(dx/2),maxy:dy:miny-(dy/2));

%reference 
Ref1=makerefmat(minx,maxy,dx,dy);
[r1,c1]=map2pix(Ref1,xii,yii);
Z=nan(length(xii(:,1)),length(xii(1,:)));
r1=floor(r1);
c1=floor(c1);
ind1=sub2ind(size(Z),r1,c1);
demgrid=X1(ind1);
demgrid(demgrid<=0)=0.5;

%extract dem values
Ref2=makerefmat(minx,maxy,dx,dy);
[r2,c2]=map2pix(Ref2,x,y);
Z=nan(length(xii(:,1)),length(xii(1,:)));
r2=floor(r2);
c2=floor(c2);
ind2=sub2ind(size(Z),r2,c2);
demgrid(ind2)=dem_wells;

%extract dem values at intersection
Ref3=makerefmat(minx,maxy,dx,dy);
[r3,c3]=map2pix(Ref3,x1,y1);
Z=nan(length(xii(:,1)),length(xii(1,:)));
r3=floor(r3);
c3=floor(c3);
ind_t=sub2ind(size(Z),r3,c3);
int_dem=demgrid(ind_t);

%identify the active cells
[r3,c3]=find(X==1);
indc1=sub2ind(size(Z),r3,c3);

% dem_all=repmat(dem_wells,1,length(z(1,:)));
% zdem=(dem_all-z);
% zdem(zdem<0)=0.1;
% mzdem=min(zdem,[],2);
% mzdem1=repmat(mzdem,1,size(zdem,2));
% azdem=zdem-mzdem1;
% 

% for i=1:length(z(1,:))
%     for j=1:length(z(:,1))
%         if dem_all(j,i)-z(j,i)<=0
%             wl(j,i)=dem_all(j,i);
%         else
%             wl(j,i)=dem_all(j,i)-z(j,i);
%         end
%     end
% end

% avgzdem=mean(wl,2);

for j=1:29
    for i=1:12
    avgzdem_mon_mean(j,i)=mean(wl(j,i:12:end));
%     a=wl(j,i:12:end);
    end
end

 
% stage1
for i=1:12
% Bs=zeros(2,1);
mats=[dem_wells avgzdem_mon_mean(:,i)];
mats1=mats;
[Bs,bints,rs,rints,statss]=regress(mats1(:,end),[ones(length(mats1(:,1)),1) mats1(:,1)]);
% [Bs1,devs1,statss1{i}]=glmfit([mats1(:,1)],mats1(:,end),'normal');
 statss2{i}=regstats(mats1(:,end),[mats1(:,1)]);
% Bs=mats1(:,2)\([mats1(:,1)])
awl_grid=nan(length(xii(:,1)),length(xii(1,:)));
awl_grid_reg=Bs(1)+(Bs(2)*demgrid);
awl_grid(ind1)=awl_grid_reg(ind1);
awl_grid(awl_grid<=0)=0.5;
awl_grid(ind2)=avgzdem_mon_mean(:,i);
awl_grid1(:,:,i)=awl_grid;

end

% %checking avgzdem values
% awl_grid_act=awl_grid(ind2);
% dem_grid_act=demgrid(ind2);
% 
% avgzdem=awl_grid(ind2);
% dem=demgrid(ind2);






% stage 2 
for i=237
wlt=wl(:,i);
mi=mod(i,12);
if mi==0
    mi=12;
end
mat=[x y avgzdem_mon_mean(:,mi) wlt];
awl_grid2=awl_grid1(:,:,mi);

for k=1:1
    mat1=mat;
%     t_ind=[20,22,27,29];
%     mat1(t_ind,:)=[]; 
     mat1(k,:)=[];  
    [B,bint,r,rint,stats]=regress(mat1(:,end),[ones(length(mat1(:,1)),1) mat1(:,3)]);
     [B1,dev1,stats1]=glmfit([mat1(:,3)],mat1(:,end));
     stats2=regstats(mat1(:,end),[mat1(:,3)]);
     
    figure(100)
    set(gca,'FontSize',16)
    scatter(mat1(:,3),mat1(:,end),'ko','MarkerFaceColor','k')
    xlabel('Mean groundwater level (m)')
    ylabel('Hydraulic heads (m)')
    h1 = lsline;
   set(h1,'linewidth',2,'color','r');
    xl=get(gca,'Xlim');
yl=get(gca,'Ylim');
ml=min(xl,yl);
ml1=max(xl,yl);
ml2=[ml(1),ml1(2)];
xpos=xl(1);
ypos=yl(2);
    rho=corr(mat1(:,3),mat1(:,end));
    text(xpos+1,ypos-1.5,sprintf('Corr=%.2f',rho),'fontsize',16)

%    validation x y  
%    xv=x(ind);
%    yv=y(ind);
%    demv=dem(ind);
%    zdemv=wt(ind);
%    zpred(k,1)=b(1)+b(2)*xv+b(3)*yv+b(4)*demv;
%    zobs(k,1)=zdemv; 
   zpred_all=B(1)+B(2)*mat1(:,3);
   zobs_all=mat1(:,end);
   r_all=zobs_all-zpred_all;
   res=r_all;
   varres=var(res);
   
  
%   CR=[];
%  for m=1:length(res)
%      for n=1:length(res)
%         CR(m,n)=res(m)*res(n);        
%      end
%  end
res_v=res.^2;
CR=diag(res_v);
  %hat matrix fommation
 
%    Xmat=[ones(length(mat1(:,1)),1) mat1(:,1:4)];
%    Yvec=[mat1(:,end)];
%    
%    px=pinv((Xmat'*Xmat));
%    px1=Xmat*px;
%    px2=px1*Xmat';
%    
%    %OLS parameter esat
%    pols=pinv(Xmat'*Xmat)*Xmat'*Yvec;
%    pmvreg=mvregress(Xmat,Yvec,'algorithm','cwls')
%    
% 
%    %covarinace matrix
%    I=eye(length(res));
%    px5=varres*(I-px2);
% 
%    %GLS parameter esat
%    px6=pinv(Xmat'*pinv(px5)*Xmat);
%    px7=px6*Xmat'*pinv(px5)*Yvec;
%    
%    % check the error
%    cores=res*res';
%    
%    %crosscheck with gls estimate in amtlab
%    p1=pinv(Xmat'*pinv(px5)*Xmat)*Xmat'*pinv(px5)*Yvec;
   

   xv=x(k);
   yv=y(k);
%    demv=dem_wells(k);
% % %    wltmv=wltm(k);
% % %    ref1v=ref1(k);
%    avgzdemv=avgzdem_mon_mean(k);
%    r_allv=r_all(k);
%    
%    zpred_val(k,1)=B(1)+B(2)*xv+B(3)*yv+B(4)*demv+B(5)*avgzdemv;
%    zobs_val(k,1)=avgzdemv;
%    
%    zpred_val(k,1)=p1(1)+p1(2)*xv+p1(3)*yv+p1(4)*demv+p1(5)*avgzdemv;
%    zobs_val(k,1)=avgzdemv;
%    comp=[zobs_val zpred_val];
%    figure(100)
%    plo
   
%    +B(5)*avgzdemv;
   %all predictions
%    awl_all=B(1)+B(2)*xii+B(3)*yii+B(4)*dem_grid;
   
   
% mat2=[x y r_all];
x1=mat1(:,1);
y1=mat1(:,2);
r_all1=r_all;
d=variogram([x1 y1],r_all1,'nrbins',20);
dis=d.distance;
%dis=dis/1000
val=d.val;

a0=1500;
c0=2;
% figure(3)
[~,~,~,S]=variogramfit(dis,val,a0,c0,[],...
                       'model','spherical','solver','fminsearchbnd','nugget',0,...
                       'plotit',false);

sill=S.sill;
range=S.range;
range=range/1000;
dis=S.h;
dis=dis/1000;
gamma=S.gamma;
gammahat=S.gammahat;

figure(1)
scatter(dis,val,'*');hold on;
plot(dis,gammahat,'k')
set(gca,'FontSize',16)
xlabel('Lag distance (km)','fontweight','bold','fontsize',16)
ylabel('Semivariance (m^2)','fontweight','bold','fontsize',16)
% title('RK-MGWL semivariogram (pre-monsoon)','fontweight','bold','fontsize',16)
xl=get(gca,'Xlim');
yl=get(gca,'Ylim');
% ml=min(xl,yl);
% ml1=max(xl,yl);
% ml2=[ml(1),ml1(2)];
xpos=xl(1);
ypos=yl(2);
% text(xpos+5,ypos-5,sprintf('Exponential model',[]))
text(xpos+1,ypos-0.2,sprintf('Sill=%.2f m^2',sill),'fontsize',16)
text(xpos+1,ypos-0.5,sprintf('Range=%.2f km',range),'fontsize',16)
% text(xpos+5,ypos-5,sprintf('Sill=%f',sill))
legend({'Experimental data','Spherical model'})



%create meshgrid
% minx=min(x);
% maxx=max(x);
% miny=min(y);
% maxy=max(y);
% dx=200;dy=200;
% [xii,yii]=meshgrid(minx-dx:dx:maxx+dx,maxy+dy:-dy:miny-dy);
%  
% figure(5)
% plot(xii,yii,'c',xii',yii','c',x1,y1,'ro')

% minx=R(1,1);
% maxx=R(2,1);
% miny=R(1,2);
% maxy=R(2,2);
% dx=200;dy=-200;
% [xii,yii]=meshgrid(minx:dx:maxx-dx,maxy:dy:miny-dy);

sizest = size(xii);
numest = numel(xii);
numobs = numel(x1);

% force column vectors
xi = xii(:);
yi = yii(:);
x2  = x1(:);
y2  = y1(:);
z3 = r_all1(:);
%covarinace residual
% CR=cov(z3);
chunksize = 100;
Dx = hypot(bsxfun(@minus,x2,x2'),bsxfun(@minus,y2,y2'));
A = S.func([S.range S.sill],Dx);
Lind1=find(Dx==0);
A(Lind1)=0;
Lind2=find(Dx>S.range);
A(Lind2)=S.sill;
A1 = [[A ones(numobs,1)];ones(1,numobs) 0];
A2 = pinv(A1);
% A1=S.func([S.range S.sill],Dx);
% A1=pinv(A);
z3  = [z3;0];
zi = nan(numest,1);
s2zi = nan(numest,1);

nrloops   = ceil(numest/chunksize);
h  = waitbar(0,'Kr...kr...kriging');
kv=[];
for r = 1:nrloops
    % waitbar 
    waitbar(r / nrloops,h);    
    % built chunks
    if r<nrloops
        IX = (r-1)*chunksize +1 : r*chunksize;
    else
        IX = (r-1)*chunksize +1 : numest;
        chunksize = numel(IX);
    end
    
    % build b
    b = hypot(bsxfun(@minus,x2,xi(IX)'),bsxfun(@minus,y2,yi(IX)'));
    % again set maximum distances to the range
%     b1 = hypot(bsxfun(@minus,x2,xi(IX)'),bsxfun(@minus,y2,yi(IX)'));
%     Rind=find(b(b>S.range));
    % expand b with ones
    b1 = [S.func([S.range S.sill],b)];
    Rind1=find(b==0);
    Rind2=find(b>S.range);
    b1(Rind1)=0;
    b1(Rind2)=S.sill;
    b2=[b1;ones(1,chunksize)];
%     b1(Rind)=S.sill;
%     b1= [S.func([S.range S.sill],b1)];

     
    % solve system
    lambda = A2*b2;
%   lambda1=pinv(A1)*b1;
    % estimate zi
    zi(IX)  = lambda'*z3;
    
    % calculate kriging variance
    s2zi(IX) = sum(b2.*lambda,1);  
    %covariance at sampled locations C(h)
    cb=sill-b1;
    res_var1=cb'*pinv(CR)*cb;
    res_var2 =sill-res_var1;
%     res_var3 =diag(res_var2);
%     res_var3(IX)=res_var2
%     sill=S.sill;
%     Ch=A1;
 
%     Ch=sill-Ch;
%     %covariance at prediction location C(0)
%     C0=b1;
%     C0=sill-C0;
%     kv1=(diag((C0'*pinv(px5))*C0));
%     kv=[kv;res_var3];
   IX1=IX';
   q=[ones(length(mat1(:,1)),1) mat1(:,3)];
   q0=[ones(length(xi(IX1)),1) awl_grid2(IX1)];
   dv1=q0*(pinv(q'*pinv(CR)*q))*q0';
   dv2=diag(dv1);
   kv=[kv;dv2];   
end

%close waitbar
close(h)
zi = reshape(zi,sizest);
s2zi1 = reshape(s2zi,sizest);
trend_var=reshape(kv,sizest);
%variance from drift
%    sill=S.sill;
%    Ch=sill-Ch;
%    q=[ones(length(mat1(:,1)),1) mat1(:,1) mat1(:,2) mat1(:,3) mat1(:,4)];
%    q0=[ones(length(xii(:)),1) xii(:) yii(:) demgrid(:) awl_grid2(:)];
%    dv1=q0*(pinv(q'*pinv(CR)*q))*q0';
% %    dv2=(dv1*q0')*q0;
%    dv2=diag(dv1);
%total variance
% tred_var=reshape(dv2,sizest);
% % res_var=reshape(kv,sizest);
tvzi=reshape(s2zi1,sizest)+reshape(trend_var,sizest);
% figure(6)
% imagesc(xii(1,:),yii(:,1),zi); axis image; axis xy
% title('kriging predictions')
% figure(7)
% contour(xii,yii,s2zi); axis image
% title('kriging variance')

% MLR=B(1)+B(2)*xii+B(3)*yii+B(4)*demgrid+B(5)*awl_grid2;
% TP=MLR+zi;

%refernecing
% x11 = minx;  % Two meters east of the upper left corner
% y11 = maxy;  % Two meters south of the upper left corner
% dx =200;
% dy =-200;
% R = makerefmat(x11, y11, dx, dy);
% [r,c]=map2pix(R,x2,y2);
% Z=nan(length(xii(:,1)),length(xii(1,:)));
% r=round(r);
% c=round(c);
% indc=sub2ind(size(Z),r,c);
% ztt=zi(indc);

%validation
Ref2=makerefmat(minx,maxy,dx,dy);
[rv,cv]=map2pix(Ref2,xv,yv);
Zv=nan(length(xii(:,1)),length(xii(1,:)));
rv=floor(rv);
cv=floor(cv);
indv=sub2ind(size(Zv),rv,cv);
zv=zi(indv);
% zact=r_all(k,1);
rv1(k,1)=rv;
cv1(k,1)=cv;
MLR=B(1)+B(2)*awl_grid2;
TP=MLR+zi;


ZMLR(k,1)=MLR(indv);
Zzv(k,1)=zv;

zobs(k,1)=wlt(k,1);
zpred(k,1)=TP(indv);

%mae
% e=zv-zact;
% n=length(e);
% rmse=sqrt(sumsqr(e)/n);

%store
% wl_ts(:,:,i)=zi;
% avg_ts(i,1)=mean(zi(:));
end


end
e=zobs-zpred;
n=length(e);
mse=(sumsqr(e)/n);
me=sum(e)/n;
mae=sum(abs(e))/n;
rmse=sqrt(sumsqr(e)/n);
[r2,rmse1]=rsquare(zobs,zpred)

indices=[me mse mae rmse r2]

% int_wl=TP(ind_t);
% obs_wl=wlt([29,27,22,20]);
% 
% pred_loc=D1([3,17,52,80]);

% figure(6)
% set(gca,'FontSize',12)
% contour(xii(1,:)/1000,yii(:,1)/1000,TP,'ShowText','on'); 
% % axis image; axis xy
% title('RK-MWL water level predictions (pre monsoon)');
% xlabel('Easting(km)');
% ylabel('Northing(km)');
% colorbar('southoutside')
% figure(7)
% contour(xii(1,:),yii(:,1),demgrid-TP,'ShowText','on'); axis image; axis xy
% colorbar
% title('kriging predictions')
% 
% 
% figure(9)
% imagesc(xii(1,:),yii(:,1),TP); axis image; axis xy
% title('kriging predictions')
% colorbar
% 
% figure(10)
% imagesc(xii(1,:),yii(:,1),demgrid-TP); axis image; axis xy
% title('kriging predictions')
% colorbar

% figure(11)
% contour(xii(1,:),yii(:,1),s2zi,'ShowText','on'); axis image; axis xy
% title('kriging predictions')
% colorbar
% % 
% figure(12)
% set(gca,'FontSize',12)
% contour(xii(1,:)/1000,yii(:,1)/1000,tvzi,'ShowText','on','LevelStep',0.5); 
% % axis image; axis xy
% title('RK-MWL total error variance (pre monsoon)');
% xlabel('Easting(km)');
% ylabel('Northing(km)');
% colorbar('southoutside')
% % colormap('gray')
% 
% figure(13)
% set(gca,'FontSize',12)
% contour(xii(1,:)/1000,yii(:,1)/1000,zi,'ShowText','on','LevelStep',0.5); 
% % axis image; axis xy
% title('RK-MWL trend error (pre monsoon)');
% xlabel('Easting(km)');
% ylabel('Northing(km)');
% colorbar('southoutside')
% mesh(peaks)
% colormap('gray')


%create meshgrid


minx=R1(1,1);
maxx=R1(2,1);
miny=R1(1,2);
maxy=R1(2,2);
dx=90;dy=90;
[xii1,yii1]=meshgrid(minx:dx:maxx-(dx/2),miny:dy:maxy-(dy/2));
yii2=round(yii(:,1)/1000);


Zv(indc1)=TP(indc1);
% figure(14)
% % set(gca,'Fontsize',16)
% a=imagesc(xii1(1,:)/1000,((yii1(:,1)/1000)),(TP));
% set(gca,'Fontsize',16)
% % set(gca,'Ydir','reverse')
% % colorbar
% title('RK-MGWL water level predictions (pre-monsoon)','fontweight','bold','fontsize',16);
% xlabel('Easting (km)','fontweight','bold','fontsize',16);
% ylabel('Northing (km)','fontweight','bold','fontsize',16);
% h1=colorbar('fontsize',16)
% colormap(jet(12))
% h1.Label.String = 'Groundwater head (m)';
% caxis([0 70])
% set(a,'alphadata',~isnan(Zv))
% % set(gca,'Yticklabel',[round(linspace(yii2(1),yii2(end),6))])
% box on;
% cb = title(h1,'m');
% pos=get(cb,'position')
% pos(1) = pos(1)+45;
% pos(2)=pos(2)-3.5
% set(cb, 'position', pos);


Zv(indc1)=tvzi(indc1);
figure(15)
% set(gca,'Fontsize',16)
a=imagesc(xii1(1,:)/1000,((yii1(:,1)/1000)),(tvzi));
set(gca,'Fontsize',16)
% set(gca,'Ydir','reverse')
% colorbar
% title('RK-MGWL total error variance (pre-monsoon)','fontweight','bold','fontsize',16);
xlabel('Easting (km)','fontweight','bold','fontsize',16);
ylabel('Northing (km)');
h2=colorbar('fontsize',16)
colormap(jet(12))
ylabel(h2,'Error variance (m^2)');
% h2.Label.String = 'error variance (m^2)';
caxis([0 6])
set(a,'alphadata',~isnan(Zv))
set(gca,'Yticklabel',[round(linspace(yii2(1),yii2(end),6))])
box on;  
axis square
% cb = title(h2,'m^2');
% pos=get(cb,'position')
% pos(1) = pos(1)+28.5;
% pos(2)=pos(2)-3.5
% set(cb, 'position', pos);

Zv(indc1)=zi(indc1);
figure(16)
% set(gca,'Fontsize',16)
a=imagesc(xii1(1,:)/1000,((yii1(:,1)/1000)),(zi));
set(gca,'Fontsize',16)
% set(gca,'Ydir','reverse')
% colorbar
title('RK-MGWL trend error (pre-monsoon)','fontweight','bold','fontsize',16);
xlabel('Easting (km)','fontweight','bold','fontsize',16);
ylabel('Northing (km)','fontweight','bold','fontsize',16);
h3=colorbar('fontsize',16)
colormap(jet(12))
h3.Label.String = 'trend error (m)';
caxis([-6 6])
set(a,'alphadata',~isnan(Zv))
set(gca,'Yticklabel',[round(linspace(yii2(1),yii2(end),6))])
box on;
% cb = title(h3,'m');
% pos=get(cb,'position')
% pos(1) = pos(1)+2.5;
% pos(2)=pos(2)-3.5
% set(cb, 'position', pos);


figure(12)
set(gca,'FontSize',16)
scatter(zobs,zpred,'k','fill','o');hold on
% scatter(pred1(:,9),pred1(:,13),[],'r','s')
% scatter(pred1(:,17),pred1(:,21),[],'b','fill','<')
% scatter(pred1(:,25),pred1(:,29),[],'m','fill','o')
line([0,40],[0,40],...
          'linewidth',2,...
          'color',[1,0,0]);
%       axis square;
      xlabel('Observed water level (m)','fontweight','bold','fontsize',16)
      ylabel('Predicted water level (m)','fontweight','bold','fontsize',16)
      title('Pre-monsoon','fontweight','bold','fontsize',16)
      legend('RK-MGWL')
      box on
      
      figure(25)

scatter(mat1(:,2),mat1(:,1),'k','fill','o');hold on
set(gca,'FontSize',16)
% plot(zobs,p11,'m--')
% scatter(pred1(:,9),pred1(:,13),[],'r','s')
% scatter(pred1(:,17),pred1(:,21),[],'b','fill','<')
% scatter(pred1(:,25),pred1(:,29),[],'m','fill','o')
line([0,40],[0,40],...
          'linewidth',2,...
          'color',[1,0,0]);
%       axis square;
      xlabel('Observed groundwater heads (m)','fontweight','bold','fontsize',16)
      ylabel('MGWL (m)','fontweight','bold','fontsize',16)
      title('RK-MGWL trend component (Pre-monsoon)','fontweight','bold','fontsize',16)
      h1=lsline;
      set(h1,'linewidth',2)
      box on

%       figure(13)
%       plot(D1,int_dem,'k',D1,int_wl,'r-',pred_loc,obs_wl,'g*')
%       set(gca,'YLim',[0, 55])
      

figure(26)
% set(gca,'Fontsize',16)
a=imagesc(xii1(1,:)/1000,((yii1(:,1)/1000)),(awl_grid1(:,:,11)))
set(gca,'Fontsize',16)
% set(gca,'Ydir','reverse')
% colorbar
title('June','fontweight','bold','fontsize',16);
xlabel('Easting (km)','fontweight','bold','fontsize',16);
ylabel('Northing (km)','fontweight','bold','fontsize',16);

h1=colorbar('fontsize',16)
colormap(jet(12))
h1.Label.String = 'Groundwater head (m)';
caxis([0 70])
set(a,'alphadata',~isnan(Zv))
% set(gca,'Yticklabel',[round(linspace(yii2(1),yii2(end),6))])
box on;    

figure(27)
% set(gca,'Fontsize',16)
a=imagesc(xii1(1,:)/1000,((yii1(:,1)/1000)),(X1))
set(gca,'Fontsize',16)
% set(gca,'Ydir','reverse')
% colorbar
title('DEM','fontweight','bold','fontsize',16);
xlabel('Easting (km)','fontweight','bold','fontsize',16);
ylabel('Northing (km)','fontweight','bold','fontsize',16);
h1=colorbar('fontsize',16)
colormap(jet(12))
h1.Label.String = 'elevation (m)';
caxis([0 70])
set(a,'alphadata',~isnan(Zv))
% set(gca,'Yticklabel',[round(linspace(yii2(1),yii2(end),6))])
box on;  

figure(28)
% set(gca,'Fontsize',16)
a=imagesc(xii1(1,:)/1000,((yii1(:,1)/1000)),(awl_grid1(:,:,12)))
set(gca,'Fontsize',16)
% set(gca,'Ydir','reverse')
% colorbar
title('December','fontweight','bold','fontsize',16);
xlabel('Easting (km)','fontweight','bold','fontsize',16);
ylabel('Northing (km)','fontweight','bold','fontsize',16);

h11=colorbar('fontsize',16)
colormap(jet(12))
h11.Label.String = 'MGWL (m)';
caxis([0 70])
set(a,'alphadata',~isnan(Zv))
% set(gca,'Yticklabel',[round(linspace(yii2(1),yii2(end),6))])
box on;  

figure(14)
% set(gca,'Fontsize',16)
a=imagesc(xii1(1,:)/1000,((yii1(:,1)/1000)),(TP));
set(gca,'Fontsize',16)
% set(gca,'Ydir','reverse')
% colorbar
% title('RK-MGWL predictions (pre-monsoon)','fontweight','bold','fontsize',16);
xlabel('Easting (km)','fontweight','bold','fontsize',16);
ylabel('Northing (km)','fontweight','bold','fontsize',16);
h1=colorbar('fontsize',16)
colormap(jet(12))
ylabel(h1,'Groundwater hydraulic head (m)')
h1.Label.String = 'Groundwater head (m)';
caxis([0 70])
set(a,'alphadata',~isnan(Zv))
set(gca,'Yticklabel',[round(linspace(yii2(1),yii2(end),6))])
box on;  
axis square
      