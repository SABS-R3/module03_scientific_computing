

im = double(imread('world-map.gif'));  % read image
im=sum(im,3);                          % sum across the three colour channels
imagesc(im);                           % display the image



[v,d]=eig(cov(im));


x=im;
  m=mean(x,2);
  for i=1:size(x,1)
    x(i,:) = x(i,:) - m(i);
  end


  p=4;
    % reconstruct the data with top p eigenvectors
    y = (x*v(:,end-p+1:end))*v(:,end-p+1:end)';
    % plotting
    figure
    subplot(3,1,1);imagesc(im)
    subplot(3,1,2);imagesc(y)
    % add mean
    for i=1:size(x,1)
      y(i,:) = y(i,:) + m(i);
    end
    subplot(3,1,3);imagesc(y)



im = double(imread('Queen_Victoria_by_Bassano.jpg'));
  im = sum(im,3);

  % reconstruct the data with top p eigenvectors
  [v,d]=eig(cov(im));
  p=4;
  x=im;
  m=mean(x,2);
  for i=1:size(x,1)
    x(i,:) = x(i,:) - m(i);
  end
  y = (x*v(:,end-p+1:end))*v(:,end-p+1:end)';
  % plotting
  figure
  colormap(gray);
  subplot(1,3,1); imagesc(im);
  subplot(1,3,2); imagesc(y);
  % add mean
  for i=1:size(x,1)
  y(i,:) = y(i,:) + m(i);
  end
  subplot(1,3,3); imagesc(y)

