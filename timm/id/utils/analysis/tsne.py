import torch
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use('Agg')
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
global ax
def mscatter(x, y, m, c, **kw):
    """
    画不同形状的图
    :param x: 数据x，x[:,0]
    :param y: 数据y, x[:,1]
    :param m: 形状分类
    :param c: 预测值y,用来分类颜色
    :param kw: 其它参数
    :return:
    """
    import matplotlib.markers as mmarkers
    ax = plt.gca()
    sc = ax.scatter(x, y, c=c, **kw)
    m = list(map(lambda x: m[x], c))
    if (m is not None) and (len(m) == len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc
def mscatter_3D(x, y, z, m, c, **kw):
    """
    画不同形状的图
    :param x: 数据x，x[:,0]
    :param y: 数据y, x[:,1]
    :param m: 形状分类
    :param c: 预测值y,用来分类颜色
    :param kw: 其它参数
    :return:
    """
    import matplotlib.markers as mmarkers
    ax = plt.gca()
    sc = ax.scatter(x, y, z, c=c, **kw)
    m = list(map(lambda x: m[x], c))
    if (m is not None) and (len(m) == len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc
def visualize(source_feature: torch.Tensor, target_feature: torch.Tensor,
              source_label: torch.Tensor, target_label: torch.Tensor,
              filename: str, source_color='r', target_color='b'):
    """
    Visualize features from different domains using t-SNE.

    Args:
        source_feature (tensor): features from source domain in shape :math:`(minibatch, F)`
        target_feature (tensor): features from target domain in shape :math:`(minibatch, F)`
        filename (str): the file name to save t-SNE
        source_color (str): the color of the source features. Default: 'r'
        target_color (str): the color of the target features. Default: 'b'

    """
    # 只需要显示前plot_only个
    plot_num = 5000
    plot_only_s = min(plot_num,len(source_feature))
    plot_only_t = min(plot_num,len(source_feature))
    source_feature = source_feature[:plot_only_s].numpy().reshape(plot_only_s,-1)
    target_feature = target_feature[:plot_only_t].numpy().reshape(plot_only_t,-1)
    source_label = source_label[:plot_only_s].numpy()
    target_label = target_label[:plot_only_t].numpy()
    features = np.concatenate([source_feature, target_feature], axis=0)
    #print(features.size)

    # map features to 2-d using TSNE
    X_tsne = TSNE(n_components=2,init='pca', random_state=33).fit_transform(features)
    #Y_tsne = TSNE(n_components=2,init='pca', random_state=33).fit_transform(target_feature)
    # domain labels, 1 represents source while 0 represents target
    domains = np.concatenate((np.ones(len(source_feature)), np.zeros(len(target_feature	 ))))

    # visualize using matplotlib
    fig=plt.figure(figsize=(10, 10))
    ax = fig.gca()
    m = {0: 'o', 1: 'x', 2: 's', 3:'v',4:'1',5:'2',6:'3',7:'4',8:'8',9:'p',10:'*',11:'h',12:'H',13:'+',14:'D',15:'d'}  # 枚举所有分类想要的形状
    s1=mscatter(X_tsne[:plot_only_s, 0], X_tsne[:plot_only_s, 1], m=m, c=source_label, cmap=col.ListedColormap([source_color]), s=8)
    s2=mscatter(X_tsne[plot_only_s:, 0], X_tsne[plot_only_s:, 1], m=m, c=target_label, cmap=col.ListedColormap([target_color]), s=8)
    plt.title("features") 
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    #ax.set_zlabel('Z Label')
    plt.legend((s1,s2),('finger','spoof') ,loc = 'best')
    #plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=domains, cmap=col.ListedColormap([source_color, target_color]), s=3)    
    plt.savefig(filename)
def visualize_3D(source_feature: torch.Tensor, target_feature: torch.Tensor,
              source_label: torch.Tensor, target_label: torch.Tensor,
              filename: str, source_color='r', target_color='b'):
    """
    Visualize features from different domains using t-SNE.

    Args:
        source_feature (tensor): features from source domain in shape :math:`(minibatch, F)`
        target_feature (tensor): features from target domain in shape :math:`(minibatch, F)`
        filename (str): the file name to save t-SNE
        source_color (str): the color of the source features. Default: 'r'
        target_color (str): the color of the target features. Default: 'b'

    """
    # 只需要显示前plot_only个
    plot_num = 5000
    plot_only_s = min(plot_num,len(source_feature))
    plot_only_t = min(plot_num,len(source_feature))
    source_feature = source_feature[:plot_only_s].numpy()
    target_feature = target_feature[:plot_only_t].numpy()
    source_label = source_label[:plot_only_s].numpy()
    target_label = target_label[:plot_only_t].numpy()
    features = np.concatenate([source_feature, target_feature], axis=0)
    #print(features.size)

    # map features to 2-d using TSNE
    X_tsne = TSNE(n_components=3,init='pca', random_state=33).fit_transform(features)
    #Y_tsne = TSNE(n_components=3,init='pca', random_state=33).fit_transform(target_feature)
    # domain labels, 1 represents source while 0 represents target
    domains = np.concatenate((np.ones(len(source_feature)), np.zeros(len(target_feature))))

    # visualize using matplotlib
    fig=plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    m = {0: 'o', 1: 'x', 2: 's'}  # 枚举所有分类想要的形状
    s1=mscatter_3D(X_tsne[:plot_only_s, 0], X_tsne[:plot_only_s, 1], X_tsne[:plot_only_s, 2], m=m, c=source_label, cmap=col.ListedColormap([source_color]), s=8, alpha=0.5)
    s2=mscatter_3D(X_tsne[plot_only_s:, 0], X_tsne[plot_only_s:, 1], X_tsne[plot_only_s:, 2], m=m, c=target_label, cmap=col.ListedColormap([target_color]), s=8, alpha=0.5)
    plt.title("features") 
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.legend((s1,s2),('finger','spoof') ,loc = 'best')
    #plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=domains, cmap=col.ListedColormap([source_color, target_color]), s=3)    
    plt.savefig(filename)
def visualize_single(source_feature: torch.Tensor, 
              source_label: torch.Tensor, 
              filename: str, source_color='r'):
    """
    Visualize features from different domains using t-SNE.

    Args:
        source_feature (tensor): features from source domain in shape :math:`(minibatch, F)`
        target_feature (tensor): features from target domain in shape :math:`(minibatch, F)`
        filename (str): the file name to save t-SNE
        source_color (str): the color of the source features. Default: 'r'
        target_color (str): the color of the target features. Default: 'b'

    """
    # 只需要显示前plot_only个
    plot_num = 5000
    plot_only_s = min(plot_num,len(source_feature))
    #plot_only_t = min(plot_num,len(source_feature))
    source_feature = source_feature[:plot_only_s].numpy().reshape(plot_only_s,-1)
    #target_feature = target_feature[:plot_only_t].numpy().reshape(plot_only_t,-1)
    source_label = source_label[:plot_only_s].numpy()
    #target_label = target_label[:plot_only_t].numpy()
    features = np.concatenate([source_feature], axis=0)
    #print(features.size)

    # map features to 2-d using TSNE
    X_tsne = TSNE(n_components=2,init='pca', random_state=33).fit_transform(features)
    #Y_tsne = TSNE(n_components=2,init='pca', random_state=33).fit_transform(target_feature)
    # domain labels, 1 represents source while 0 represents target
    #domains = np.concatenate((np.ones(len(source_feature)), np.zeros(len(target_feature ))))

    # visualize using matplotlib
    fig=plt.figure(figsize=(10, 10))
    ax = fig.gca()
    m = {0: 'o', 1: 'x', 2: 's'}  # 枚举所有分类想要的形状
    s1=mscatter(X_tsne[:plot_only_s, 0], X_tsne[:plot_only_s, 1], m=m, c=source_label, cmap=col.ListedColormap([source_color]), s=8)
    #s2=mscatter(X_tsne[plot_only_s:, 0], X_tsne[plot_only_s:, 1], m=m, c=target_label, cmap=col.ListedColormap([target_color]), s=8)
    plt.title("features") 
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    #ax.set_zlabel('Z Label')
    plt.legend('finger','spoof' ,loc = 'best')
    #plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=domains, cmap=col.ListedColormap([source_color, target_color]), s=3)    
    plt.savefig(filename)