function h1 = plot_variance(x,lower,upper,color)
    h1 = fill([x,x(end:-1:1)],[upper,fliplr(lower)],color);
    set(h1,'EdgeColor','none');
    plot(x,upper,'color',color);
    plot(x,lower,'color',color);
end