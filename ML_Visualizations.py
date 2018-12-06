import matplotlib
import matplotlib.pyplot as plt
import numpy


def make_color(clrs, hnew,gnum):

    c = numpy.array([0,0,0], dtype=numpy.float)

    colors = numpy.array(clrs, dtype=numpy.float)

    for j in range(len(hnew[0])):
        c += colors[j]*hnew[gnum][j]*.5
    return list(c.tolist())


def calculate_m_b(x1, y1, x2, y2):
    m = (y2-y1)/(x2-x1)
    b = y1 - m*x1
    return m, b


def calculate_y(x, m, b):
    return m*x + b


def calculate_x(y, m, b):
    return (y - b)/m


def line_calc_x(x1, y1, x2, y2, new_y):
    m , b = calculate_m_b(x1, y1, x2, y2)
    x = calculate_x(new_y, m, b)
    return int(numpy.around(x, 0))


def make_scree_graph_data(np_data_array, show_it=True):
    u, s, vh = numpy.linalg.svd(np_data_array, full_matrices=True, compute_uv=True)
    v = numpy.transpose(vh)

    sum_s = sum(s.tolist())
    s_sum = numpy.cumsum(s)[-1]
    print('shape of s')
    print(s.shape)
    obs_var = np_data_array.shape
    num_obs = obs_var[0]
    num_var = obs_var[1]

    print('There are {:d} observations and {:d} variables or attributes'.format(num_obs, num_var))

    eigen_vals = s ** 2 /s_sum

    single_vals = numpy.arange(num_obs)

    fig = plt.figure(figsize=(8, 5))
    plt.plot(single_vals, eigen_vals, 'ro-', linewidth=2)
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Eigenvalue')
    # I don't like the default legend so I typically make mine like below, e.g.
    # with smaller fonts and a bit transparent so I do not cover up data, and make
    # it moveable by the viewer in case upper-right is a bad place for it
    leg = plt.legend(['Eigenvalues from SVD'], loc='best', borderpad=0.3,
                     shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                     markerscale=0.4)
    leg.get_frame().set_alpha(0.4)
    leg.draggable(state=True)

    if show_it:
        plt.show()
    return u, s, vh, v


def make_scree_plot_usv(s, num_vars, show_it=True, last=False, k_val=3):
    sum_s = sum(s.tolist())
    s_sum = numpy.cumsum(s**2)[-1]

    eigen_vals = (s ** 2) / s_sum

    single_vals = numpy.arange(num_vars)

    kret = 0

    oldp = -900

    for i in range(1, num_vars-1):
        if numpy.around((eigen_vals[i-1] - eigen_vals[i]), 2) == 0:
        #if (eigen_vals[i-1] - eigen_vals[i]) == 0:
        #print('prev', eigen_vals[i-1])
        #print('curnt', eigen_vals[i])
        #print('---------',eigen_vals[i]/eigen_vals[i-1])
        #crnt = (eigen_vals[i]/eigen_vals[i-1])
        #if crnt < oldp:
            #if (eigen_vals[i] - eigen_vals[i-1]) < .05:
            kret = i
            break

        #oldp = crnt

    print('k for sckree tis ', kret)

    if show_it:
        fig = plt.figure(figsize=(8, 5))
        plt.plot(single_vals, eigen_vals, 'ro-', linewidth=2)
        plt.title('Scree Plot')
        plt.xlabel('Principal Component')
        plt.ylabel('Eigenvalue')

        index_k = list(single_vals.tolist()).index(k_val)

        print('k_val',k_val)

        #plt.plot(k_val, eigen_vals[index_k], 'go')
        plt.plot(k_val, eigen_vals[k_val], 'go')

        plt.plot(kret, eigen_vals[kret], 'bo')

        leg = plt.legend(['# of PC\'s vs. Eigen values','k from POV','elbow estimate'], loc='best',
                         borderpad=0.3,shadow=False,
                         prop=matplotlib.font_manager.FontProperties(size='small'),
                         markerscale=0.4)

        leg.get_frame().set_alpha(0.4)
        leg.draggable(state=True)


    if last:
        plt.show()
    return kret+1


def make_prop_o_var_plot(s, num_obs, show_it=True, last_plot=True):

    sum_s = sum(s.tolist())

    ss = s**2

    sum_ss = sum(ss.tolist())

    prop_list = list()

    found = False

    k = 0

    x1, y1, x2, y2, = 0,0,0,0
    p_l, i_l = 0, 0
    found = False

    for i in range(1, num_obs+1):
        perct = sum(ss[0:i]) / sum_ss
        #perct = sum(s[0:i]) / sum_s

        if numpy.around((perct*100), 0) >= 90 and not found:
            y2 = perct
            x2 = i
            x1 = i_l
            y1 = p_l
            found = True
        prop_list.append(perct)
        i_l = i
        p_l = perct

    if numpy.around(y2, 2) == .90:
        k_val = x2
    else:
        print('it is over 90%',x2)
        k_val = line_calc_x(x1, y1, x2, numpy.around(y2,2), .9)

    single_vals = numpy.arange(1,num_obs+1)

    if show_it:
        fig = plt.figure(figsize=(8, 5))
        plt.plot(single_vals, prop_list, 'ro-', linewidth=2)
        plt.title('Proportion of Variance, K should be {:d}'.format(x2))
        plt.xlabel('Eigenvectors')
        plt.ylabel('Prop. of var.')

        p90 = prop_list.index(y2)

        #plt.plot(k_val, prop_list[p90], 'bo')
        plt.plot(x2, prop_list[p90], 'bo')

        leg = plt.legend(['Eigenvectors vs. Prop. of Var.','90% >=  variance'], loc='best', borderpad=0.3,
                         shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                         markerscale=0.4)
        leg.get_frame().set_alpha(0.4)
        leg.draggable(state=True)

        if last_plot:
            plt.show()

    return x2


def dual_scree_prop_var(s, num_obs):
    sum_s = sum(s.tolist())

    eigen_vals = s ** 2 /sum_s

    single_vals = numpy.arange(num_obs)

    prop_list = list()

    for i in range(1, num_obs + 1):
        prop_list.append(sum(s[0:i].tolist()) / sum_s)


    fig, axs = plt.subplots(2,1)
    #plt.figure(1)
    #fig = plt.figure(figsize=(8, 5))
    axs[0].plot(single_vals, eigen_vals, 'ro-',  linewidth=2)
    plt.title('Scree Plot')
    #axs[0].title('Scree Plot')
    axs[0].set_xlabel('Principal Component')
    axs[0].set_ylabel('Eigenvalue')
    leg = axs[0].legend(['Eigenvalues from SVD'], loc='best', borderpad=0.3,
                     shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                     markerscale=0.4)
    leg.get_frame().set_alpha(0.4)
    leg.draggable(state=True)

    #plt.figure(2)
    axs[1].plot(single_vals, prop_list, 'go-', linewidth=2)
    #plt.title('Proportion of Variance')
    axs[1].set_xlabel('Eigenvectors')
    axs[1].set_ylabel('Prop. of var.')
    leg = plt.legend(['Eigenvectors vs. Prop. of Var.'], loc='best', borderpad=0.3,
                     shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                     markerscale=0.4)
    leg.get_frame().set_alpha(0.4)
    leg.draggable(state=True)
    '''
    plt.subplot(2,2,1)
    #fig = plt.figure(figsize=(8, 5))
    plt.plot(single_vals, prop_list, 'ro-', linewidth=2)
    plt.title('Proportion of Variance')
    plt.xlabel('Eigenvectors')
    plt.ylabel('Prop. of var.')
    leg = plt.legend(['Eigenvectors vs. Prop. of Var.'], loc='best', borderpad=0.3,
                     shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                     markerscale=0.4)
    leg.get_frame().set_alpha(0.4)
    leg.draggable(state=True)
    '''

    plt.show()

    return


def basic_scatter_plot(x, y, x_label, y_label, title, legend):

    fig = plt.figure(figsize=(8, 5))
    plt.scatter(x, y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    leg = plt.legend([legend], loc='best', borderpad=0.3,
                     shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                     markerscale=0.4)
    leg.get_frame().set_alpha(0.4)
    leg.draggable(state=True)
    plt.show()
    return


def z_scatter_plot(Z, schools, x_label='z1', y_label='z2', title='PC1 vs. PC2 for all Observations',
                   legend='z1 vs. z2', show_it=True, last=False, point_size=20, color=[[0,0,0]]):

    if show_it:
        fig = plt.figure(figsize=(8, 5))
        i = 0
        for row in Z:
            z1 = row[0]
            z2 = row[1]
            plt.scatter(z1, z2, s=point_size, c=color)
            plt.annotate(schools.index(schools[i]), (z1, z2))
            i += 1

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        leg = plt.legend([legend], loc='best', borderpad=0.3,
                         shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                         markerscale=0.4)
        leg.get_frame().set_alpha(0.4)
        leg.draggable(state=True)

        if last:
            plt.show()
    return


def k_cluster_scatter_plot(Z, schools, mid, groups, x_label='x1', y_label='x2', title='PC1 vs. PC2 for all Observations',
                           legend='z1 vs. z2', show_it=True, colors=[[.8, .4, .2]], b_list=[] ,g_ids = {},
                           show_center=True, last=False, groups_l=[], em=False, hnew=numpy.array([]), an_type=0):


    row_mid = len(mid)

    markers_s = list(['o','^','s','*'])

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    i = 0
    for row in Z:
        z1 = row[0]
        z2 = row[1]

        if len(b_list) > 0:
            # grab group i call it l
            l = list(b_list[i].tolist())

            # look for what group this observation is a part of
            # call the group number midx
            #midx = l.index(1) % len(colors)

            grpnm = l.index(1)
            midx = grpnm % 9
            m = markers_s[grpnm%4]
            #print(l.index(1))
        '''
        if not em:
            if schools.index(schools[i]) == -22:
                ax.scatter(z1, z2, s=30, c=[255/255, 144/255, 18/255])
            else:
                ax.scatter(z1, z2, s=20, c=colors[midx])
        else:
            if schools.index(schools[i]) == -22:
                ax.scatter(z1, z2, s=30, c=[255/255, 144/255, 18/255])
            else:
                ax.scatter(z1, z2, s=20, c=colors[midx])
        '''
        #ax.scatter(z1, z2, s=20, c=colors[midx])
        ax.scatter(z1, z2, s=20, c=colors[midx], marker=m)

        if len(groups_l) > 0:
            ax.annotate(groups_l[i], (z1, z2))
        elif len(b_list[i]) > 100:
            ax.annotate(grpnm, (z1, z2))
        else:
            ax.annotate(schools.index(schools[i]), (z1, z2))
        i += 1
    if show_center:
        #r_c = b_list.shape
        #bii = list()
        i = 0
        # for row in mid:
        for row, color in zip(mid, colors):
            m1 = row[0]
            m2 = row[1]
            if len(hnew) > 0:
                gmx = numpy.max(hnew[:, i])
                gmd = numpy.median(hnew[:, i])
                gmn = numpy.min(hnew[:, i])
                glist = list([gmx, gmd,gmn])
                #glist = list([gmn, gmd,gmx])
                for m in range(len(mid)):
                    ax.scatter(m1, m2, s=3000*(glist[m]), c=color, alpha=(1/(m+1))*glist[m-1])
            else:
                ax.scatter(m1, m2, s=3000, c=color, alpha=.5)
            # ax.annotate(groups[i], (m1, m2), arrowprops=dict(facecolor='black', shrink=1.05))
            ax.annotate(groups[i], (m1, m2))
            i += 1

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    #leg = plt.legend(legend, loc='best', borderpad=0.3,
    #                 shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
    #                 markerscale=0.4, )
    #leg.get_frame().set_alpha(0.4)
    #leg.draggable(state=True)
    if last:
        plt.show()
    return
