 %water level interpolation
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
% [X2,map2,R2]=geotiffread('D:\mohanasundaram\paper6_maps\gis\images\export\soil.tif');
% X=double(X);
% X1=double(X1);


x=well(2:end,3);
x=cell2mat(x);
y=well(2:end,4);
y=cell2mat(y);
dem=well(2:end,7);
dem=cell2mat(dem)+10;
wl=well(2:end,9:end);
wl=cell2mat(wl);
wl(wl<0)=0;

dem_wells=well(2:end,7);
dem_wells=cell2mat(dem_wells);

%create meshgrid
minx=R1(1,1);
maxx=R1(2,1);
miny=R1(1,2);
maxy=R1(2,2);
dx=90;dy=-90;
[xii,yii]=meshgrid(minx:dx:maxx-(dx/2),maxy:dy:miny-(dy/2));

figure(5)
plot(xii,yii,'c',xii',yii','c',x,y,'ro')

%reference 
Ref1=makerefmat(minx,maxy,dx,dy);
[r1,c1]=map2pix(Ref1,xii,yii);
Z=nan(length(xii(:,1)),length(xii(1,:)));
r1=floor(r1);
c1=floor(c1);
ind1=sub2ind(size(Z),r1,c1);
demgrid=X1(ind1);
demgrid(demgrid<0)=0.5;
% demgrid=demgrid;
%extract dem values
Ref2=makerefmat(minx,maxy,dx,dy);
[r2,c2]=map2pix(Ref2,x,y);
Z=nan(length(xii(:,1)),length(xii(1,:)));
r2=floor(r2);
c2=floor(c2);
ind2=sub2ind(size(Z),r2,c2);
demgrid(ind2)=dem_wells;

%identify the active cells
[r3,c3]=find(X==1);
indc1=sub2ind(size(Z),r3,c3);

% dem_int=demgrid(ind2);
% dem_int(dem_int<=0)=1;
% demgrid=demgrid;
% demgrid=demgrid;
% dem=repmat(dem_int,1,length(z(1,:)));
% for i=1:length(z(1,:))
%     for j=1:length(z(:,1))
%         if dem(j,i)-z(j,i)<=0
%             wl(j,i)=dem(j,i);
%         else
%             wl(j,i)=dem(j,i)-z(j,i);
%         end
%     end
% end
% 
mat=[x,y,wl];
xt=mat(:,1);
yt=mat(:,2);
zt=mat(:,3:end);

% mat1=mat;
% ind=[21];
% %     2,12,14,8];
% mat1(ind,:)=[];
% x=mat1(:,1);
% y=mat1(:,2);
% z=mat1(:,3:end);
% 
% %validation x y
% xv=xt(ind);
% yv=yt(ind);

for i=237
    
   for k=1:1
   mat1=mat;
    ind=k;
%     %    2,12,14,8];
    mat1(ind,:)=[];
   x=mat1(:,1);
   y=mat1(:,2);
   z=mat1(:,3:end);

    %validation x y  
   xv=xt(ind);
   yv=yt(ind);

    zf=zeros(length(zt(:,1)),1);
    for j=1:length(zt(:,1))
    zf(j,1)=zt(j,i);
    end
    
    z1=zeros(length(z(:,1)),1);
    for j=1:length(z(:,1))
    z1(j,1)=z(j,i);
    end
    z2=z1;
   
% well_id=extractfield(well,'WELL_NO');
x=x;
y=y;
z3=z2; 
% z1(1:1,1)=0
x1=[x y];

% d=variogram(x1,z3,'nrbins',20);
% dis=d.distance;
% %dis=dis/1000
% val=d.val;

a0=14000;
c0=60;
% [z3, lamda] = boxcox(z3_1);
% z3_n1 = normalize(z3_n);
d=variogram(x1,z3,'nrbins',20);
dis=d.distance;
%dis=dis/1000
val=d.val;

% figure(3)
[~,~,~,S]=variogramfit(dis,val,a0,c0,[],...
                       'model','spherical','solver','fminsearchbnd',...
                       'plotit',false);

%normalise




% figure(1)
% hist(z3)
% 
% figure(2)
% qqplot(z3)
% 
% [nh,p] = kstest(z3)
 
 
figure(3);
sill=S.sill;
range=S.range;
range=range/1000;
dis=S.h;
dis=dis/1000;
gamma=S.gamma;
gammahat=S.gammahat;
scatter(dis,val,'*');hold on
set(gca,'FontSize',16)
plot(dis,gammahat,'k')
xlabel('Lag distance (km)','fontweight','bold','fontsize',16)
ylabel('Semivariance (m^2)','fontweight','bold','fontsize',16)
% title('OK semivariogram (pre-monsoon)','fontweight','bold','fontsize',16)
xl=get(gca,'Xlim');
yl=get(gca,'Ylim');
ml=min(xl,yl);
ml1=max(xl,yl);
ml2=[ml(1),ml1(2)];
xpos=ml2(1);
ypos=ml2(2);
% text(xpos+5,ypos-5,sprintf('Exponential model',[]))
text(xpos+1,ypos-5,sprintf('Sill=%.2f m^2',sill),'fontsize',16)
text(xpos+1,ypos-10,sprintf('Range=%.2f km',range),'fontsize',16)
% text(xpos+5,ypos-5,sprintf('Sill=%f',sill))
legend({'Experimental data','Spherical model'})
% create meshgrid
% minx=min(x);
% maxx=max(x);
% miny=min(y);
% maxy=max(y);
% dx=200;dy=200;
% [xii,yii]=meshgrid(minx-dx:dx:maxx+dx,maxy+dy:-dy:miny-dy);
%  
% figure(5)
% plot(xii,yii,'c',xii',yii','c',x,y,'ro')

sizest = size(xii);
numest = numel(xii);
numobs = numel(x);

% force column vectors
xi = xii(:);
yi = yii(:);
x  = x(:);
y  = y(:);
z3 = z3(:);

chunksize = 100;
Dx = hypot(bsxfun(@minus,x,x'),bsxfun(@minus,y,y'));
A = S.func([S.range S.sill],Dx);
Lind1=find(Dx==0);
A(Lind1)=0;
Lind2=find(Dx>S.range);
A(Lind2)=S.sill;
A1 = [[A ones(numobs,1)];ones(1,numobs) 0];
A2 = pinv(A1);
% A1=pinv(A1);
z3  = [z3;0];
zi = nan(numest,1);
s2zi = nan(numest,1);

nrloops   = ceil(numest/chunksize);
h  = waitbar(0,'kriging...');
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
    b = hypot(bsxfun(@minus,x,xi(IX)'),bsxfun(@minus,y,yi(IX)'));
    % again set maximum distances to the range
    
    b1 = [S.func([S.range S.sill],b)];
    Rind1=find(b==0);
    Rind2=find(b>S.range);
    b1(Rind1)=0;
    b1(Rind2)=S.sill;
    b2=[b1;ones(1,chunksize)];
    
%     b1 = hypot(bsxfun(@minus,x,xi(IX)'),bsxfun(@minus,y,yi(IX)'));
%     
%     % expand b with ones
%     b = [S.func([S.range S.sill],b);ones(1,chunksize)];
%     
%     b1= [S.func([S.range S.sill],b1)];
%  
    
    % solve system
    lambda = A2*b2;
    
    % estimate zi
    zi(IX)  = lambda'*z3;
    
    % calculate kriging variance
    s2zi(IX) = sum(b2.*lambda,1);
    sillv=repmat(S.sill,1,length(IX));
    
    %covariance at sampled locations C(h)
    Ch=A1;
    %covariance at prediction location C(0)
    C0=b2;
    kv1=diag((C0'*pinv(Ch))*C0);
    kv=[kv;kv1];
%   (pinv(A1).*C0).*C0';
%   kv(IX)=sillv-(;

%transform data
%     zi_1 = ((lamda.*zi)+1).^(1./lamda);
%     s2zi_1 = ((lamda.*s2zi)+1).^(1./lamda);


end

%close waitbar
close(h)
zi = reshape(zi,sizest);
s2zi = reshape(s2zi,sizest);
kv=reshape(kv,sizest);
% figure(6)
% imagesc(xii(1,:),yii(:,1),zi); axis image; axis xy
% title('kriging predictions')
% figure(7)
% contour(xii,yii,s2zi); axis image
% title('kriging variance')


%refernecing
% x11 = minx;  % Two meters east of the upper left corner
% y11 = maxy;  % Two meters south of the upper left corner
% % dx =200;
% % dy =-200;
% R = makerefmat(x11, y11, dx, dy);
% [r,c]=map2pix(R,x,y);
% Z=nan(length(xii(:,1)),length(xii(1,:)));
% r=round(r);
% c=round(c);
% indc=sub2ind(size(Z),r,c);
% ztt=zi(indc);

%validation
Ref1=makerefmat(minx,maxy,dx,dy);
[rv,cv]=map2pix(Ref1,xv,yv);
Zv=nan(length(xii(:,1)),length(xii(1,:)));
rv=round(rv);
cv=round(cv);
indv=sub2ind(size(Zv),rv,cv);
zv=zi(indv);
zact=zf(ind,1);

zpred(k,1)=zv;
zobs(k,1)=zact;
%mae
% e=zv-zact;
% n=length(e);
% rmse=sqrt(sumsqr(e)/n);

%store
wl_ts(:,:,i)=zi;
avg_ts(i,1)=mean(zi(:));

dtw=demgrid-zi;
% diff=demgrid-hyd
% dtw(dtw<0)=0;
% [rf,cf]=find(dtw<0);
% indf=sub2ind(size(dtw),rf,cf);
% dtw(indf)=0.5;
% hyd=demgrid-dtw;
    end
end
% save('wl_ts','wl_ts')
% figure(6)
% set(gca,'FontSize',16)
% contour(xii(1,:)/1000,yii(:,1)/1000,zi,'ShowText','on');
% % axis image; axis xy
% % title('OK water level predictions (pre-monsoon)','fontweight','bold','fontsize',16);
% xlabel('Easting (km)','fontweight','bold','fontsize',16);
% ylabel('Northing (km)','fontweight','bold','fontsize',16);
% colorbar('southoutside','fontsize',16)

% figure(8)
% contour(xii(1,:),yii(:,1),s2zi,'ShowText','on'); axis image; axis xy
% title('kriging predictions')
% colorbar

% figure(9)
% set(gca,'FontSize',16)
% contour(xii(1,:)/1000,yii(:,1)/1000,s2zi,'ShowText','on'); 
% % axis image; axis xy
% % title('OK error variance (pre-monsoon)','fontweight','bold','fontsize',16);
% xlabel('Easting (km)','fontweight','bold','fontsize',16);
% ylabel('Northing (km)','fontweight','bold','fontsize',16);
% colorbar('southoutside','fontsize',16)


%
%create meshgrid
minx=R1(1,1);
maxx=R1(2,1);
miny=R1(1,2);
maxy=R1(2,2);
dx=90;dy=90;
[xii1,yii1]=meshgrid(minx:dx:maxx-(dx/2),miny:dy:maxy-(dy/2));
% Z1=Z;
% s2zi1=flipud(s2zi);
% Z(indc1)=s2zi(indc1);
% Z=nan(length(xii(1,:)),length(xii(:,1)));
Zv(indc1)=s2zi(indc1);

yii2=round(yii(:,1)/1000);


% figure(10)
% % set(gca,'Fontsize',16)
% a=imagesc((xii1(1,:)/1000),((yii1(:,1)/1000)),(s2zi));
% set(gca,'Fontsize',16)
% % set(gca,'Ydir','reverse')
% % colorbar
% % title('OK total error variance (pre-monsoon)','fontweight','bold','fontsize',16);
% xlabel('Easting (km)','fontweight','bold','fontsize',16);
% ylabel('Northing (km)','fontweight','bold','fontsize',16);
% h1=colorbar('fontsize',16)
% ylabel(h1,'Error variance (m^2)');
% set(a,'alphadata',~isnan(Zv))
% colormap(jet(12))
% % h1.Label.String = 'error variance (m^2)';
% caxis([0 6])
% set(a,'alphadata',~isnan(Zv))
% set(gca,'Yticklabel',[round(linspace(yii2(1),yii2(end),6))])
% box on;
% cb = title(h1,'m^2');
% pos=get(cb,'position')
% pos(1) = pos(1)+28;
% pos(2)=pos(2)-3.5
% set(cb, 'position', pos);

% set(gca,'Yticklabel',[round(yii(:,1)/1000)])
% 
% 
% Zv(indc1)=zi(indc1);
figure(10)
% set(gca,'Fontsize',16)
a=imagesc((xii1(1,:)/1000),((yii1(:,1)/1000)),(s2zi));
set(gca,'Fontsize',16)
% set(gca,'Ydir','reverse')
% colorbar
% title('OK total error variance (pre-monsoon)','fontweight','bold','fontsize',16);
xlabel('Easting (km)','fontweight','bold','fontsize',16);
ylabel('Northing (km)','fontweight','bold','fontsize',16);
h1=colorbar('fontsize',16)
ylabel(h1,'Error variance (m^2)');
set(a,'alphadata',~isnan(Zv))
colormap(jet(12))
% h1.Label.String = 'error variance (m^2)';
caxis([0 30])
set(a,'alphadata',~isnan(Zv))
set(gca,'Yticklabel',[round(linspace(yii2(1),yii2(end),6))])
set(gca,'Box','on');
axis square
% cb = title(h1,'m^2');
% pos=get(cb,'position')
% pos(1) = pos(1)+28;
% pos(2)=pos(2)-3.5
% set(cb, 'position', pos);

% set(gca,'Yticklabel',[round(yii(:,1)/1000)])
% 
% 
% Zv(indc1)=zi(indc1);
figure(11)
% set(gca,'Fontsize',16)
a=imagesc(xii1(1,:)/1000,((yii1(:,1)/1000)),(zi));
set(gca,'Fontsize',16)
% set(gca,'Ydir','reverse')
% colorbar
% title('OK predictions (pre-monsoon)','fontweight','bold','fontsize',16);
xlabel('Easting (km)','fontweight','bold','fontsize',16);
ylabel('Northing (km)','fontweight','bold','fontsize',16);
h2=colorbar('fontsize',16)
ylabel(h2,'Groundwater hydraulic head (m)')
set(a,'alphadata',~isnan(Zv))
set(gca,'Yticklabel',[round(linspace(yii2(1),yii2(end),6))])
colormap(jet(12))
% h2.Label.String = 'Groundwater head (m)';
caxis([0 70])
set(gca,'Box','on');
axis square


% cb = title(h2,'m');
% pos=get(cb,'position')
% pos(1) = pos(1)+17;
% pos(2)=pos(2)-3.5
% set(cb, 'position', pos);

% set(get(cb,'title'),'string','Dislocation Density(m^{-2})')
% lbpos = get(cb,'title'); 
% set(handle,'Position',[v(1)+15 v(4)-15]);





% Z2=Z;
% Z(indc1)=zi(indc1);
% figure(11)
% % set(gca,'Fontsize',16)
% a=imagesc(xii(:,1)/1000,yii(end,:)/1000,Z);
% set(gca,'Fontsize',16)
% % colorbar
% title('OK water level predictions (pre monsoon)','fontweight','bold','fontsize',16);
% xlabel('Easting(km)','fontweight','bold','fontsize',16);
% ylabel('Northing(km)','fontweight','bold','fontsize',16);
% colorbar('southoutside','fontsize',16)
% set(a,'alphadata',~isnan(Z))
% box on;

% figure(10)
% set(gca,'FontSize',12)
% contour(xii(1,:)/1000,yii(:,1)/1000,kv,'ShowText','on','LevelStep',5); 
% % axis image; axis xy
% title('OK pre monsoon error variance');
% xlabel('Easting(km)');
% ylabel('Northing(km)');
% colorbar('southoutside')

% figure(9)
% imagesc(xii(1,:),yii(:,1),zi); axis image; axis xy
% title('kriging predictions')
% colorbar
% 
% figure(10)
% imagesc(xii(1,:),yii(:,1),dtw); axis image; axis xy
% title('kriging predictions')
% colorbar
% 
% figure(11)
% imagesc(xii(1,:),yii(:,1),s2zi); axis image; axis xy
% title('kriging predictions')
% colorbar
% 
% figure(12)
% contour(xii(1,:),yii(:,1),s2zi); axis image; axis xy
% title('kriging predictions')
% colorbar
% 
e=zobs-zpred;
n=length(e);
mse=(sumsqr(e)/n);
me=sum(e)/n;
mae=sum(abs(e))/n;
rmse=sqrt(sumsqr(e)/n);
[r2,rmse1]=rsquare(zobs,zpred);

indices=[me mse mae rmse r2]

% figure(12)
% set(gca,'FontSize',16)
% scatter(zobs,zpred,'k','fill','o');hold on
% % scatter(pred1(:,9),pred1(:,13),[],'r','s')
% % scatter(pred1(:,17),pred1(:,21),[],'b','fill','<')
% % scatter(pred1(:,25),pred1(:,29),[],'m','fill','o')
% line([0,35],[0,35],...
%           'linewidth',2,...
%           'color',[1,0,0]);
% %       axis square;
%       xlabel('Observed water level (m)','fontweight','bold','fontsize',16)
%       ylabel('Predicted water level (m)','fontweight','bold','fontsize',16)
%       title('Pre-monsoon','fontweight','bold','fontsize',16)
%       legend('OK')
%       box on
% figure(7)
% plot(zobs);hold on;
% plot(zpred,'r')
% ae=abs(zpred-zobs)/n