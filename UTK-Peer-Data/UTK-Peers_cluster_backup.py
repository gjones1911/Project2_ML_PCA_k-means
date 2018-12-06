from Create_UTK_Data_files import *
from DimensionReduction import x_to_z_projection_pca
from k_means_cluster import *
import sys

# check for command line argurments and process accordingly
if len(sys.argv) != 1 and len(sys.argv) != 4:
    p(len(sys.argv))
    p('usage: no command line arguments prints everything including:')
    p('                                                            All Groupings')
    p('                                                            Scree and POV plots, ')
    p('                                                            cluster plots for original')
    p('                                                            cluster plot for PCA ')
    p('                                                            cluster plots PCA with 2 PC\'s')
    p('                                                            cluster plots PCA with  Estimation Maximization')
    p('usage: show_groupings(y/n) show_scree_POV(y/n) show_clusters(y/n)')
    quit(1)
if len(sys.argv) == 4 and sys.argv[1].lower() != 'y' and sys.argv[1].lower() != 'n':
    p(len(sys.argv))
    p(1)
    p('usage: no command line arguments prints everything including:')
    p('                                                            All Groupings')
    p('                                                            Scree and POV plots, ')
    p('                                                            cluster plots for original')
    p('                                                            cluster plot for PCA ')
    p('                                                            cluster plots PCA with 2 PC\'s')
    p('                                                            cluster plots PCA with  Estimation Maximization')
    p('usage: show_groupings(y/n) show_scree_POV(y/n) show_clusters(y/n)')
    quit(2)
if len(sys.argv) == 4 and sys.argv[2].lower() != 'y' and sys.argv[2] != 'n':
    p(2)
    p('usage: no command line arguments prints everything including:')
    p('                                                            All Groupings')
    p('                                                            Scree and POV plots, ')
    p('                                                            cluster plots for original')
    p('                                                            cluster plot for PCA ')
    p('                                                            cluster plots PCA with 2 PC\'s')
    p('                                                            cluster plots PCA with  Estimation Maximization')
    p('usage: show_groupings(y/n) show_scree_POV(y/n) show_clusters(y/n)')
    quit(3)
if len(sys.argv) == 4 and sys.argv[3].lower() != 'y' and sys.argv[3].lower() != 'n':
    p(3)
    p('usage: no command line arguments prints everything including:')
    p('                                                            All Groupings')
    p('                                                            Scree and POV plots, ')
    p('                                                            cluster plots for original')
    p('                                                            cluster plot for PCA ')
    p('                                                            cluster plots PCA with 2 PC\'s')
    p('                                                            cluster plots PCA with  Estimation Maximization')
    p('usage: show_groupings(y/n) show_scree_POV(y/n) show_clusters(y/n)')
    quit(4)

# my initial guess for k
km = 7

# ---------------------------------------------- PART 1 --------------------------------------------------------------
# ---------------------------------------------- Part1-1 -------------------------------------------------------------
# Load  a data matrix out of given data
# loaded needed data files for utk data
# default loads the utk data
data_list = load_data_files()

# un pack the data
utk_label = data_list[0]        # attrib labels for data
utk_data = data_list[1]         # data as a list of lists
np_utk_data = data_list[2]      # data as a numpy array for higher level calculations
imputated_data = data_list[3]   # avg imputated data
s_name = data_list[4]           # list of the names of the schools
head_l = data_list[5]           # list of headers?
attribs = data_list[6]          # a list where the rows are the attributes probably redundant
stats = data_list[7]            # an array of list containing various stats, unpacked below

num_obs = len(s_name)           # grab the number of observations
# print('here',num_obs)
mu_a = stats[0]                 # an array of the means of the variables/attributes
std_a = stats[1]                # an array of the stand. deviations  of the variables/attributes
min_a = stats[2]                # an array of the minimums of the variables/attributes
max_a = stats[3]                # an array of the maximums of the variables/attributes

# ------------------------------------- Part1-2  -----------------------------------------------------------------------
# use numpy to perform spectral vector decomposition for PCA
u, s, vh = numpy.linalg.svd(np_utk_data, full_matrices=True, compute_uv=True)
# U *Sig*Vt = X
v = numpy.transpose(vh)

vx = v[:, 0]
vy = v[:, 1]

print('s shape', s.shape)
print('here 2',np_utk_data.shape)
# use the prop of variance graphing function to get a good estimate of k

# ------------------------------------- Part1-3 -------------------------------------------------------------------
# -------------------------------------plot a scree graph of singular values and % of variance using (s)
# -------------------------------------use this to guess a good value for k
if len(sys.argv) == 4 and sys.argv[2].lower() == 'y' and sys.argv[3] == 'y':
    p('a')
    p(sys.argv[2])
    k = make_prop_o_var_plot(s, num_obs, show_it=True, last_plot=False)
    make_scree_plot_usv(s, num_obs, show_it=True, last=False)
    p('The original k:')
    p('The best k for original data is {:d}'.format(k))
    km = k
elif len(sys.argv) == 4 and sys.argv[2].lower == 'y' and sys.argv[3] == 'n':
    p('b')
    k = make_prop_o_var_plot(s, num_obs, show_it=True, last_plot=False)
    make_scree_plot_usv(s, num_obs, show_it=True, last=True)
    p('The original k:')
    p('The best k for original data is {:d}'.format(k))
    km = k
elif len(sys.argv) == 4 and sys.argv[2].lower() == 'n':
    p('c')
    k = make_prop_o_var_plot(s, num_obs, show_it=False, last_plot=False)
    p('The original k:')
    p('The best k for original data is {:d}'.format(k))
    km = k
else:
    k = make_prop_o_var_plot(s, num_obs, show_it=True, last_plot=False)
    make_scree_plot_usv(s, num_obs, show_it=True, last=False)
    p('The original k:')
    p('The k for original data is {:d}'.format(km))
    # km = k
# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

# -------------------------------------------- Part1-4 ----------------------------------------------------------------
ko = km

# grab the first k principle components vectors
W = v[:, 0:k]
WT = numpy.transpose(W)

# grab the first two principle components for part 1-5
W2 = v[:, 0:2]
WT2 = numpy.transpose(W2)
# perform the projection using the principle components from above
z_array = x_to_z_projection_pca(WT, np_utk_data, numpy.array(mu_a, dtype=numpy.float))
z_array2 = x_to_z_projection_pca(WT2, np_utk_data, numpy.array(mu_a, dtype=numpy.float))

'''
p('')
p('')
p('The shape of z array is:')
print(z_array.shape)
p('The shape of z array with 2 PC\'s is:')
print(z_array2.shape)
p('')
p('')
'''
# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------- Part1-5 ---------------------------------------------------------------
# set up the scatter plot of the data
z_title = 'The first 2 PC for the transformed data into {:d} dimensional space'.format(k)
z_scatter_plot(z_array, s_name, title=z_title, last=False, show_it=True)
# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------

# --------------------------------------------- PART 2 ---------------------------------------------------------------


# --------------------------------------------- Part 2-6,7,8,9,10,11,12 -----------------------------------------------
# do k means clustering using original data
'''
found = False
while not found:
    try:
        #rcc, init_mk = make_rand_m(np_utk_data, km)
        #end_mk,  iter, bi_l = k_means_clustering(np_utk_data, km, init_m=init_mk)
        #end_mk,  iter_n, bi_l = k_means_clustering(np_utk_data, km, m_init_type=2, mu_a=mu_a, min_a=min_a,
        #                                           max_a=max_a)
        end_mk,  iter_n, bi_l = k_means_clustering(np_utk_data, km)
        grps = create_group_l(list(bi_l.tolist()), km)
        # use the mid points to do some analysis so we can project them into the z plane later
        um, sm, vhm = numpy.linalg.svd(end_mk, full_matrices=True, compute_uv=True)

        # make sure I found the number of groups I wanted
        found = check_grouping(grps)

        mid = min_intercluster_distance(np_utk_data, grps)
        mxid = max_intracluster_distance(np_utk_data, grps)
        p('The number of interations: {:d}'.format(iter_n))
        p('The min intercluster distance is : {:f}'.format(mid))
        p('The max intracluster distance is : {:f}'.format(mxid))
        dun_o = mid/mxid
        p('the dun index is for k-means of original data is: {:f}'.format(dun_o))
    except numpy.linalg.LinAlgError:
        found = False
        print('we have and error')
'''

end_mk, iter_n, bi_l, grps, vhm , dun_o = k_means_processor(np_utk_data, km)
p('')
p('')

# do k means clustering using dimension reduced data data
'''
found = False
while not found:
    try:
        #init_mk = make_given_m(z_array, rcc)
        #end_mk2,  iter2, bi_l2 = k_means_clustering(z_array, km, init_m=init_mk)
        #end_mk2,  iter2, bi_l2 = k_means_clustering(z_array_n, km)
        #z_stats = get_basic_stats(z_array)
        #end_mk2,  iter2, bi_l2 = k_means_clustering(z_array, km, m_init_type=2, mu_a=z_stats[0], min_a=z_stats[2],
        #                                            max_a=z_stats[3])
        end_mk2,  iter2, bi_l2 = k_means_clustering(z_array, km)
        grps2 = create_group_l(list(bi_l2.tolist()), km)
        # end_mk,  iter, bi_l = k_means_clustering(np_utk_data, km, mu_a, min_a, max_a)
        um2, sm2, vhm2 = numpy.linalg.svd(end_mk2, full_matrices=True, compute_uv=True)
        found = check_grouping(grps2)
        mid = min_intercluster_distance(z_array, grps2)
        mxid = max_intracluster_distance(z_array, grps2)
        dun_z = mid/mxid
        p('The number of interations: {:d}'.format(iter2))
        p('The min intercluster distance is : {:f}'.format(mid))
        p('The max intracluster distance is : {:f}'.format(mxid))
        p('the dun for projected data/k={:d}index is: {:f}'.format(k, dun_z))
    except numpy.linalg.LinAlgError:
        found = False
        print('we have and error')
'''
end_mk2, iter2, bi_l2, grps2, vhm2, dun_z = k_means_processor(z_array, km)

p('')
p('')

'''
found = False
# cluster useing the first two PC's only
while not found:
    try:
        #z2_stats = get_basic_stats(z_array2)
        #end_mk3,  iter3, bi_l3 = k_means_clustering(z_array2, km, m_init_type=2, mu_a=z2_stats[0], min_a=z2_stats[2],
        #                                            max_a=z2_stats[3])
        end_mk3,  iter3, bi_l3 = k_means_clustering(z_array2, km)
        grps3 = create_group_l(list(bi_l3.tolist()), km)
        # end_mk,  iter, bi_l = k_means_clustering(np_utk_data, km, mu_a, min_a, max_a)
        um3, sm3, vhm3 = numpy.linalg.svd(end_mk3, full_matrices=True, compute_uv=True)
        found = check_grouping(grps3)
        mid = min_intercluster_distance(z_array2, grps3)
        mxid = max_intracluster_distance(z_array2, grps3)
        dun_z2 = mid/mxid
        p('The number of interations: {:d}'.format(iter3))
        p('The min intercluster distance is : {:f}'.format(mid))
        p('The max intracluster distance is : {:f}'.format(mxid))
        p('the dun index for 1st 2 PC is: {:f}'.format(dun_z2))
    except numpy.linalg.LinAlgError:
        found = False
        print('we have and error')
'''

end_mk3, iter3, bi_l3, grps3, vhm3, dun_z2 = k_means_processor(z_array2, km)

p('')
p('')
# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------

# -------------------------------------------- Part 2-13 ----------------------------------------------------------
# attempt to perform expectation maximization
# mnew, hnew = expectation_maximization(np_utk_data, end_mk, bi_l)
# mnew, hnew = expectation_maximization(z_array, end_mk2, bi_l2)
mnew, hnew, em_itera = expectation_maximization(z_array2, end_mk3, bi_l3)
uem, sem, vhem = numpy.linalg.svd(mnew, full_matrices=True, compute_uv=True)
# make a grouping
gbb = get_EM_grouping(hnew, km)
grps4 = create_group_l(list(gbb.tolist()), km)


em_mid = min_intercluster_distance(z_array2, grps4)
em_mxid = max_intracluster_distance(z_array2, grps4)
p('The number of interations: {:d}'.format(em_itera))
p('The min intercluster distance is : {:f}'.format(em_mid))
p('The max intracluster distance is : {:f}'.format(em_mxid))
dun_em = em_mid / em_mxid
p('the dun index is for k-means of original data is: {:f}'.format(dun_em))

show_group = False

# ----------------Grouping Display-----------------------------------------------------
if len(sys.argv) == 1 and show_group:
    p('----------------------------------------------------------------------------------')
    p('Original Data Clustering:')
    show_grouping(grps)
    p('----------------------------------------------------------------------------------')
    p('')
    p('----------------------------------------------------------------------------------')
    p('PCA  Data Clustering:')
    show_grouping(grps2)
    p('----------------------------------------------------------------------------------')
    p('')
    p('----------------------------------------------------------------------------------')
    p('First 2 PC\'s Clustering:')
    show_grouping(grps3)
    p('----------------------------------------------------------------------------------')
    p('')
    p('----------------------------------------------------------------------------------')
    p('EM data Clustering:')
    show_grouping(grps4)
    p('----------------------------------------------------------------------------------')
    p('')

elif len(sys.argv) > 1:
    if sys.argv[1] == 'y':
        p('----------------------------------------------------------------------------------')
        p('Original Data Clustering:')
        show_grouping(grps)
        p('----------------------------------------------------------------------------------')
        p('')
        p('----------------------------------------------------------------------------------')
        p('PCA  Data Clustering:')
        show_grouping(grps2)
        p('----------------------------------------------------------------------------------')
        p('')
        p('----------------------------------------------------------------------------------')
        p('First 2 PC\'s Clustering:')
        show_grouping(grps3)
        p('----------------------------------------------------------------------------------')
        p('')
        p('----------------------------------------------------------------------------------')
        p('EM data Clustering:')
        show_grouping(grps4)
        p('----------------------------------------------------------------------------------')
        p('')
    # ----------------------------------------------------------------------

vm = numpy.transpose(vhm)
vem = numpy.transpose(vhem)
vm2 = numpy.transpose(vhm2)
vm3 = numpy.transpose(vhm3)

#vmx = vm[:, 0]
#vmy = vm[:, 1]

#k = make_prop_o_var_plot(sm, km, show_it=True, last_plot=False)

print('The value of K should be {:d}'.format(k))

Wm = vm[:, 0:k]

WTm = numpy.transpose(Wm)

m_stats2 = get_basic_stats(end_mk2)
m_stats3 = get_basic_stats(end_mk3)

uem, sem, vhem = numpy.linalg.svd(mnew, full_matrices=True, compute_uv=True)
Wem = vhem[0:k]
Wtem = numpy.transpose(Wem)
m_em_stat = get_basic_stats(mnew)
#mid_points_em = x_to_z_projection_pca(Wtem, mnew, numpy.array(m_em_stat[0], dtype=numpy.float))

m_stats = get_basic_stats(end_mk)
# mid_points = x_to_z_projection_pca(WTm, end_mk, numpy.array(m_stats[0], dtype=numpy.float))
mid_points = project_mid_points(end_mk, vm, km)
mp2 = project_mid_points(end_mk2, vm2, km)
mp3 = project_mid_points(end_mk3, vm3, km)
emp = project_mid_points(mnew, vem, km)

#mid_points2 = x_to_z_projection_pca(WTm2, end_mk2, numpy.array(m_stats2[0], dtype=numpy.float))
#mid_points3 = x_to_z_projection_pca(WTm3, end_mk3, numpy.array(m_stats3[0], dtype=numpy.float))

colors_a = [[1, 0, 0],      # 0-red
            [0, 1, 0],      # 1-blue
            [0, 0, 1],      # 2-green
            [0, 0, 0],      # 3-black
            [1, 1, 0],      # 4-purp
            [1, 0, 1],      # 5
            [0, 1, 1],      # 6-yellow
            [.5, .5, .5],   # 7
            [.5, .2, .1],   # 8
            [.12, .2, .8]]  # 9

groups = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

legend_titles = ['Group 1',
                 'Group 2',
                 'Group 3',
                 'Group 4']

r_l = make_label_list(30)

#for lbl in r_l:
#    p(lbl)


if len(sys.argv) == 1:
    k_cluster_title = 'K means with {:d} groups for UTK peers data, dun = {:.2f}'.format(km, dun_o)
    #z_scatter_plot(mid_points, groups, show_it=True)
    k_cluster_scatter_plot(z_array, s_name, mid_points, groups, colors=colors_a, b_list=bi_l, show_center=False,
                           title=k_cluster_title, legend=legend_titles, last=False, show_it=True)

    titlea = 'K cluster w/projected data and grouping {:d} dun is {:.2f}'.format(km, dun_z)
    k_cluster_scatter_plot(z_array, s_name, mp2, groups, colors=colors_a, b_list=bi_l2, show_center=False,
                           title=titlea, legend=legend_titles, last=False, show_it=True)

    title2 = 'K Clustering for projected data 2 PC with dun {:.2f}'.format(dun_z2)
    k_cluster_scatter_plot(z_array2, s_name, mp3, groups, colors=colors_a, b_list=bi_l3, show_center=False,
                           title=title2, legend=legend_titles, last=False, show_it=True)

    title = 'K Cluster for projected data {:d} PC w/EM'.format(ko)
    k_cluster_scatter_plot(z_array2, s_name, emp, groups, colors=colors_a, b_list=gbb, show_center=False,
                           title=title, legend=legend_titles, last=True, show_it=True, em=True, hnew=hnew)
elif sys.argv[3].lower() == 'y':
    k_cluster_title = 'K means with {:d} groups for UTK peers data, dun = {:.2f}'.format(km, dun_z)
    z_scatter_plot(mid_points, groups, show_it=False)
    k_cluster_scatter_plot(z_array, s_name, mid_points, groups, colors=colors_a, b_list=bi_l, show_center=False,
                           title=k_cluster_title, legend=legend_titles, last=False, show_it=True)

    titlea = 'K cluster w/projected data and grouping {:d} dun is {:.2f}'.format(km, dun_z)
    k_cluster_scatter_plot(z_array, s_name, mid_points, groups, colors=colors_a, b_list=bi_l2, show_center=False,
                           title=titlea, legend=legend_titles, last=False, show_it=True)

    title2 = 'K Clustering for projected data 2 PC with dun {:.2f}'.format(dun_z2)
    k_cluster_scatter_plot(z_array2, s_name, mid_points, groups, colors=colors_a, b_list=bi_l3, show_center=False,
                           title=title2, legend=legend_titles, last=False, show_it=True)

    title = 'K Cluster for projected data {:d} PC w/EM'.format(ko)
    k_cluster_scatter_plot(z_array2, s_name, mnew, groups, colors=colors_a, b_list=gbb, show_center=True,
                           title=title, legend=legend_titles, last=True, show_it=True, em=True, hnew=hnew)

