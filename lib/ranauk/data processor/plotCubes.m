function plotCubes(x,y,z,V,C)
%function plotCubes(x,y,z,V,C)
%makes 3D plot of shape defined by boolean 3d-matrix V, with the color
%the small cubes given by the similar shaped matrix C
%x, y and z are vectors indicating the edge locations, so their lengths
%should be one larger than the corresponding dimension of the matrices

[nx,ny,nz] = size(V);

if length(x) ~= nx + 1 || length(y) ~= ny + 1 || length(z) ~= nz + 1
    error('Length of x, y and z must be one larger than size of V and C')
end

dc = 0.02 * (max(C(find(V>0)))) - min(C(find(V>0))); %small color difference to create 3D effect

%-X
ilin = find(cat(1, V(1,:,:), ~V(1:end-1,:,:) & V(2:end,:,:)))';
[ix,iy,iz] = ind2sub([nx,ny,nz], ilin);
px = repmat(x(ix), 4, 1);
py = [y(iy); y(iy); y(iy+1); y(iy+1)];
pz = [z(iz); z(iz+1); z(iz+1); z(iz)];
c = C(ilin);
%-Y
ilin = find(cat(2, V(:,1,:), ~V(:,1:end-1,:) & V(:,2:end,:)))';
[ix,iy,iz] = ind2sub([nx,ny,nz], ilin);
px = [px, [x(ix); x(ix); x(ix+1); x(ix+1)]];
py = [py, repmat(y(iy), 4, 1)];
pz = [pz, [z(iz); z(iz+1); z(iz+1); z(iz)]];
c = [c, C(ilin) + dc];
%-Z
ilin = find(cat(3, V(:,:,1), ~V(:,:,1:end-1) & V(:,:,2:end)))';
[ix,iy,iz] = ind2sub([nx,ny,nz], ilin);
px = [px, [x(ix); x(ix); x(ix+1); x(ix+1)]];
py = [py, [y(iy); y(iy+1); y(iy+1); y(iy)]];
pz = [pz, repmat(z(iz), 4, 1)];
c = [c, C(ilin) - dc];
%+X
ilin = find(cat(1, V(1:end-1,:,:) & ~V(2:end,:,:), V(end,:,:)))';
[ix,iy,iz] = ind2sub([nx,ny,nz], ilin);
px = [px, repmat(x(ix+1), 4, 1)];
py = [py, [y(iy); y(iy); y(iy+1); y(iy+1)]];
pz = [pz, [z(iz); z(iz+1); z(iz+1); z(iz)]];
c = [c, C(ilin)];
%+Y
ilin = find(cat(2, V(:,1:end-1,:) & ~V(:,2:end,:), V(:,end,:)))';
[ix,iy,iz] = ind2sub([nx,ny,nz], ilin);
px = [px, [x(ix); x(ix); x(ix+1); x(ix+1)]];
py = [py, repmat(y(iy+1), 4, 1)];
pz = [pz, [z(iz); z(iz+1); z(iz+1); z(iz)]];
c = [c, C(ilin) + dc];
%+Z
ilin = find(cat(3, V(:,:,1:end-1) & ~V(:,:,2:end), V(:,:,end)))';
[ix,iy,iz] = ind2sub([nx,ny,nz], ilin);
px = [px, [x(ix); x(ix); x(ix+1); x(ix+1)]];
py = [py, [y(iy); y(iy+1); y(iy+1); y(iy)]];
pz = [pz, repmat(z(iz+1), 4, 1)];
c = [c, C(ilin) - dc];

%figure;
patch(px,py,pz,c)
axis equal vis3d
view(-50, 50);
set(gca,'FontSize',6)
shading flat; %dc can be set to zero when using faceted
clear px py pz c ilin ix iy iz nx ny nz