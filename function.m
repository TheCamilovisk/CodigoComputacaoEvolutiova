[X,Y] = meshgrid(-25:.25:25);

Z = 0.5-((((sin(sqrt(X.^2+Y.^2))).^2)-(0.5))./(((1)+(0.001).*((X.^2+Y.^2))).^2));

surf(X,Y,Z)

colormap hsv

colorbar
