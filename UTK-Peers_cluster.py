from Create_UTK_Data_files import *
from k_means_cluster import *
from  mi_cluster_tester import *
import sys


# --------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------- PART 1 --------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------


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

mx_np = numpy.array(max_a)
mn_np = numpy.array(min_a)
mu_np = numpy.array(mu_a, dtype=numpy.float64)
std_np = numpy.array(std_a, dtype=numpy.float64)

'''
data_type = 0
while data_type != 1 and data_type != 2 and data_type != 3:
    data_type = int(input('Do you want to use 1) normalized or 2) unnormalized? '))
    if data_type != 1 and data_type != 2:
        print('You must pick 1 or 2')
'''
k_type = 0

# choose to have POV plot shoose k or choose your own
while k_type < 1:
    k_type = int(input('Do you want to use k as: 1) --the k chosen by pov\n'
                       '                         any int >1 --some k chosen by you '))
    if k_type < 1:
        print('You must choose some number greater than 1')

# it they chose to choose their own k set intitial m to be a random selection

                   #use a random initial
# other wise ask it they want to use the best one from testing or a random one
m_type = -1
while 0 > m_type:
    m_type = int(input('What type of initial m do you want to use: 1) best tested m\n'
                           '                                           2) random m '))
    if 2 < m_type or m_type < 1:
        print('Choice must be 1 or 2')
        m_type = -1

print('mtype', m_type)

data_type = 3

#if data_type == 1:
#    np_utk_data = (np_utk_data-mn_np)/(mx_np-mn_np)
#elif data_type == 3:
np_utk_data = (np_utk_data - mu_np) / std_np


# ------------------------------------- Part1-2  -----------------------------------------------------------------------
# use numpy to perform spectral vector decomposition for PCA
u, s, vh = numpy.linalg.svd(np_utk_data, full_matrices=False, compute_uv=True)
# U *Sig*Vt = X
v = numpy.transpose(vh)

# -----------------------------------------------------------------------------------------------------------------
# use the prop of variance graphing function to get a good estimate of k

# ------------------------------------- Part1-3 -------------------------------------------------------------------
# -------------------------------------plot a scree graph of singular values and % of variance using (s)
# -------------------------------------use this to guess a good value for k
kpov = make_prop_o_var_plot(s, len(s), show_it=True, last_plot=False)
ks = make_scree_plot_usv(s, len(s), show_it=True, last=False, k_val=kpov)
k = kpov

if k_type == 1:
    k = kpov
    print('using the POV estimated k: {:d}'.format(k))
else:
    k = k_type
    print('using the user defined k: {:d}'.format(k))

km = k
# ------------------------------------------------------------------------------------------------------------------

# -------------------------------------------- Part1-4 ----------------------------------------------------------------
ko = km
# grab the first k principle components vectors
W = v[:, 0:k]
WT = numpy.transpose(W)

#print('shape of wt is ', W.shape)
# grab the first two principle components for part 1-5
W2 = v[:, 0:2]
WT2 = numpy.transpose(W2)

# perform the projection into k dimensional space using the principle components from above
z_array = x_to_z_projection_pca(WT, np_utk_data, mu_np)
z_array2 = x_to_z_projection_pca(WT2, np_utk_data, mu_np)

# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------- Part1-5 ---------------------------------------------------------------

# set up the scatter plot of the data

z_title = 'The scatter plot using  2 PCs for the normalized data'

#z_scatter_plot(z_array, s_name, title=z_title, last=False, show_it=False)
z_scatter_plot(z_array, s_name, title=z_title, last=False, show_it=True)
# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------

# --------------------------------------------- PART 2 ---------------------------------------------------------------


# --------------------------------------------- Part 2-6,7,8,9,10,11,12 -----------------------------------------------

#        make a random set of k initial reference variables

print('m type is ', m_type)


if m_type == 1:
    r_c , b_dun_avg = cluster_tester(np_utk_data, km, z_array, z_array2, 300)
    print('Best random m: ', r_c)
    print('best avg dun: ', b_dun_avg)
elif m_type == 2:
    r_c = numpy.random.choice(num_obs, km, replace=False)


#r_c =[1, 2, 3, 8, 12, 13, 14, 19, 21, 22, 26, 27, 34, 35, 38, 40, 46, 51, 54, 55]
#r_c = [3, 5, 12, 13, 17, 20, 21, 23, 27, 37, 46, 48, 53, 55]
#r_c = [3, 12, 14, 19, 20, 22, 26, 34, 38, 40, 42, 45, 56]
#r_c = [1, 3, 4, 9, 10, 15, 18, 22, 26, 42, 50]
#r_c = [2, 3, 7, 17, 20, 22, 35, 43, 51, 54]
#r_c = [2, 8, 14, 15, 18, 24, 34, 37, 51]
#r_c = [8, 18, 21, 22, 25, 26, 28, 30]
#r_c = [3, 22, 29, 36, 44, 49, 50]
#r_c = [3, 8, 10, 19, 45, 50]

#r_c =[22, 24, 25, 29, 55]
#r_c = [2, 19, 44]
#             k = 12
#r_c = [3, 4, 8, 15, 18, 22, 26, 31, 40, 50, 51, 52]  #1
#r_c = [0, 3, 12, 13, 19, 22, 23, 26, 36, 39, 42, 47]  #2
#r_c = [6, 8, 11, 14, 15, 22, 27, 37, 41, 42, 46, 55]  #3
#r_c = sorted([31, 39, 42, 53, 21, 37, 43, 3, 22, 8, 15, 18])  #p
#r_c = sorted([39, 42, 53, 21, 37, 43, 3, 22, 8, 15, 18, 17])  #p



#                   norml
#               k = 2, normal
#r_c = [39,18]

# z norml
#r_c = [3,47]

#               k = 3
#r_c = [3,22,23]

#       z_normal
#r_c = [19,  5, 17]
#r_c = [29,  23, 5]

#               k = 4
#r_c = [16, 22,  0, 52]

# z norml
#r_c = [24, 41, 14, 18]
#r_c = [18,2,0,24]

#               k = 5
#r_c = [3,  8, 13, 49, 44]
#r_c = [29, 19, 50, 25, 18]

#               k = 6
#r_c = [40, 22, 51, 26, 50, 25]

#               k = 7
#r_c = [43, 22, 34, 13,  6, 27, 54]

#               k = 8
#r_c = [21, 18, 27, 23, 51, 14, 46, 13]

#               k = 9
#r_c = [5, 23, 25, 27,  8, 33,  2, 56, 28]

#               k = 10
#r_c = [51,  2, 22, 29, 41,  4,  3, 44,  1, 25]


#                   unnorml
#               k = 2
#r_c = [22, 27]

#               k = 3
#r_c = [28, 31, 33]

#               k = 4
#r_c = [22,  0, 30, 26]

#               k = 5
#r_c = [8, 29, 21, 22, 51]

#               k = 6
#r_c = [

#               k = 7
#r_c = [2,8,14,15,23,32,52]

#               k = 8
#r_c = [

#               k = 9
#r_c = [

#               k = 10
#r_c = [

#            k = 12
#            z_norm
#r_c = [1, 3, 5, 6, 12, 13, 17, 22, 28, 36, 41, 44]

#r_c = [31,42,21,56,43,4,15,17,50,23,3,22]

#        Unnormalized
#           k = 3
#r_c = list([3,25,0]) # the one i like for unnorm k=4
#           k = 7
#r_c = list([3,19,4,8,51,44,55])
#           k = 6
#r_c = list([3,19,4,8,51,0]) #unnorm
#           k = 5
#r_c = list([2,5,16,50,29,39]) #unnorm
#           k = 4
#r_c = list([2,5,24,39]) #unnorm

#        Normalized
#           k = 4
#r_c = [22,15,1,31]
#r_c = [22,5,4,31]
#r_c = [22,14,24,32]
#r_c = [22,14,4,29]
#r_c = [22,5,4,31]
#           k = 3
#r_c = []
#           k = 6
#r_c = [22,14,4,56,46,55]
#           k = 7
#r_c = []
#           k = 8
#r_c = []


p('')
print('r_c:')
print(r_c)
p('')



# do k means clustering using original data
p('doing  k means clustering using original data')
end_mk, iter_n, bi_l, grps, vhm, dun_o = k_means_processor(np_utk_data, km, rc=r_c)
p('')
p('')
# do k means clustering using dimension reduced data data
p('do k means clustering using dimension reduced data')
end_mk2, iter2, bi_l2, grps2, vhm2, dun_z = k_means_processor(z_array, km, rc=r_c)
p('')
p('')
p('do k means clustering using dimension reduced data using first 2 PC\'s')
end_mk3, iter3, bi_l3, grps3, vhm3, dun_z2 = k_means_processor(z_array2, km, rc=r_c)
p('')
p('')
# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------

# -------------------------------------------- Part 2-13 ----------------------------------------------------------
# attempt to perform expectation maximization
p('Performing Expectation Maximization.......')
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

#show_group = False
show_group = True

utk_names = create_group_names(grps, s_name)
pcak_names = create_group_names(grps2, s_name)
pca2_names = create_group_names(grps3, s_name)
pcaem_names = create_group_names(grps4, s_name)

# ----------------Grouping Display-----------------------------------------------------
if show_group:
    p('----------------------------------------------------------------------------------')
    p('Original Data Clustering:')
    #show_grouping(grps)
    print('Iterations: {:d}'.format(iter_n))
    show_grouping(utk_names)
    p('----------------------------------------------------------------------------------')
    p('')
    p('----------------------------------------------------------------------------------')
    p('PCA  Data Clustering:')
    #show_grouping(grps2)
    print('Iterations: {:d}'.format(iter2))
    show_grouping(pcak_names)
    p('----------------------------------------------------------------------------------')
    p('')
    p('----------------------------------------------------------------------------------')
    p('First 2 PC\'s Clustering:')
    #show_grouping(grps3)
    print('Iterations: {:d}'.format(iter3))
    show_grouping(pca2_names)
    p('----------------------------------------------------------------------------------')
    p('')
    p('----------------------------------------------------------------------------------')
    p('EM data Clustering:')
    #show_grouping(grps4)
    print('Iterations: {:d}'.format(em_itera))
    show_grouping(pcaem_names)
    p('----------------------------------------------------------------------------------')
    p('')

# ----------------------------------------------------------------------

vm = numpy.transpose(vhm)
vem = numpy.transpose(vhem)
vm2 = numpy.transpose(vhm2)
vm3 = numpy.transpose(vhm3)

Wm = vm[:, 0:k]

WTm = numpy.transpose(Wm)

m_stats2 = get_basic_stats(end_mk2)
m_stats3 = get_basic_stats(end_mk3)

uem, sem, vhem = numpy.linalg.svd(mnew, full_matrices=True, compute_uv=True)
Wem = vhem[0:k]
Wtem = numpy.transpose(Wem)
m_em_stat = get_basic_stats(mnew)

mid_points = project_mid_points(end_mk, vm, km)
mp2 = project_mid_points(end_mk2, vm2, km)
mp3 = project_mid_points(end_mk3, vm3, km)
emp = project_mid_points(mnew, vem, km)


colors_a = [[1, 0, 0],                        # 0-red
            [0, 1, 0],                        # 1-green
            [0, 0, 1],                        # 2-blue
            [0, 0, 0],                        # 3-black
            [1, 1, 0],                        # 4-yellow
            [148/255, 18/255, 1],             # 5-purp
            [.5, .5, .5],                     # 6-grey
            [241/255, 156/255, 187/255],      # 7-pastel magenta
            [77/255, 1, 1],                   # 8-electric blue
            [170/255, 254/255, 156/255],      # 9-pale green
            [1, 102/255, 229/255],            # 10-rose pink
            [184/255, 97/255, 62/255],        # 11-copper
            [1, 140/255, 25/255],             # 12-carrot orange
            [127/255, 1, 212/255],            # 13-aquamarine
            [233/255, 214/255, 107/255],      # 14-arylide yellow
            [251/255, 206/255, 177/255],      # 15-appricot
            [161/255,202/255,241/255],        # 16-baby blue
            [61/255,12/255,2/255],            # 17-black bean
            [208/255,1,20/255],               # 18-arctic lime
            [95/255,158/255, 160/255],        # 19-cadet blue
            [135/255,169/255,107/255],        # 20-Asparagus
            [205/255,149/255,79/255],         # 21-Antique brass
            [150/255,147/255,60/255],         # 22-dark tan
            [1,75/255,49/255],                # 23-protland orange
            [109/255,24/255,12/255],          # 24-persian plum
            [109/255,106/255,11/255],         # 25-field darb
            [150/255,146/255,36/255],         # 26-UC gold
            [96/255, 150/255, 132/255],       # 27-Auro metal suarus
            [207/255,161/255,1],              # 28-mauve
            [229/255,53/255,30/255],          # 29-cg red
            [204/255, 1, 103/255],            # 30-inchworm
            [67/255,131/255,248/255],         # 31-french blue
            [226, 0, 173/255],                # 32-fushia
            [226/255,192/255,54/255],         # 33-meat brown
            [161/255,122/255,116/255]]


groups = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

legend_titles = ['Group 1',
                 'Group 2',
                 'Group 3',
                 'Group 4']

r_l = make_label_list(30)


show_graphs = True

if show_graphs:
#if len(sys.argv) == 1:

    if data_type == 1:
        k_cluster_title = 'Clustering {:d} groups for original data, dunn = {:.2f}'.format(km, dun_o)
    else:
        k_cluster_title = 'K means with {:d} groups for original data, dunn = {:.2f}'.format(km, dun_o)
    #z_scatter_plot(mid_points, groups, show_it=True)
    k_cluster_scatter_plot(z_array, s_name, mid_points, groups, colors=colors_a, b_list=bi_l, show_center=False,
                           title=k_cluster_title, legend=legend_titles, last=False, show_it=True)

    if data_type == 1:
        titlea = 'Clusters with PCA, {:d} k vectors, dunn is {:.2f}'.format(km, dun_z)
    else:
        titlea = 'Clusters using PCA and {:d} k vectors dunn is {:.2f}'.format(km, dun_z)
    k_cluster_scatter_plot(z_array, s_name, mp2, groups, colors=colors_a, b_list=bi_l2, show_center=False,
                           title=titlea, legend=legend_titles, last=False, show_it=True)

    if data_type == 1:
        title2 = 'Clusters w/ PCA, 1st 2 PC\'s,  dunn {:.2f}'.format(dun_z2)
    else:
        title2 = 'Clusters using PCA with 1st 2 PC\'s,  dunn {:.2f}'.format(dun_z2)
    k_cluster_scatter_plot(z_array2, s_name, mp3, groups, colors=colors_a, b_list=bi_l3, show_center=False,
                           title=title2, legend=legend_titles, last=False, show_it=True)

    if data_type == 1:
        title = 'Clusters w/PCA & EM, {:d} groups, dunn: {:f}'.format(ko, dun_em)
    else:
        title = 'Clusters with PCA & EM, {:d} groups, dunn: {:f}'.format(ko, dun_em)
    k_cluster_scatter_plot(z_array2, s_name, emp, groups, colors=colors_a, b_list=gbb, show_center=False,
                           title=title, legend=legend_titles, last=True, show_it=True, em=True, hnew=hnew)

