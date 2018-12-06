from Process_CSV_to_Json import*
import os


def remove_list(l, rmv):
    """Will remove the items in rmv from given list and return a new version of this
        updated list

    :param l: list to remove items from
    :param rmv: list of items to remove
    :return: a copy of the list with the items in rmv removed
    """
    ret_l = l[:]
    for el in rmv:
        if el in ret_l:
            idx = ret_l.index(el)
            ret_l = ret_l[0:idx] + ret_l[idx+1:]
    return ret_l


def write_list_to_file(list_to_write, file_name):
    """This writes the rows of the given list to the lines of the given file

    :param list_to_write: the list you want to write
    :param file_name: the file you want to write to
    :return: None
    """

    f = open(file_name, 'wt')
    cnt = 0
    for entry in list_to_write:
        if cnt < len(list_to_write):
            f.write(str(entry)+'\n')
        else:
            f.write(str(entry))
        cnt += 1

    #for row in list_to_write:
    #    for i in range(len(row)):
    #        f.write(row[i])
    #        if i < len(row)-1:
    #            f.write(' ')
    #        else:
    #            f.write('\n')

    f.close()
    return


def load_list_from_file(file_name, convert=False, convert_type='', vectorize=False, split_val=' '):
    """This will load the lines of given file into a list and return it

    :param file_name: file to load from
    :param convert: optional, Do you need to convert the info from a string
    :param convert_type: optional, The type you need to convert into
    :param vectorize: optional, Do you need to vectorize the lines
    :param split_val: optional, What you use to split the lines into vector entries
    :return: will return a list of the lines of the file, it may be a list of lists depending
    """
    f = open(file_name, 'rt')
    lines = f.readlines()
    f.close()

    def make_float_vec(l, vect=False, split_v=" "):
        if vect:
            rl = l.strip('\n').split(split_v)
            for i in range(len(rl)):

                rl[i] = float(rl[i].strip('[').strip(']').strip(',').strip(' '))
        else:
                rl = float(l)
        return rl

    def make_int_vec(l, vect=False, split_v=" "):
        if vect:
            rl = l.strip('\n').split(split_v)
            for i in range(len(rl)):
                rl[i] = int(rl[i])
        else:
                rl = int(l)
        return rl

    ret_l = list()

    for l in lines:
        if vectorize:
            if convert:
                if convert_type == 'int':
                    ret_l.append(make_int_vec(l, vectorize, split_val))
                elif convert_type == 'float':
                    ret_l.append(make_float_vec(l, vectorize, split_val))
            else:
                rl = l.strip('\n').split(split_val)
                for i in range(len(rl)):
                    ret_l.append(rl)
        else:
            ret_l.append(l)

    return ret_l


def load_data_from_file(file_name, convert=True, convert_type='float', vectorize=True, split_val=' ', label=True):
    """This will load the lines of given file into a list and return it

    :param file_name: file to load from
    :param convert: optional, Do you need to convert the info from a string
    :param convert_type: optional, The type you need to convert into
    :param vectorize: optional, Do you need to vectorize the lines
    :param split_val: optional, What you use to split the lines into vector entries
    :param label: optional, are there labels (T/F)
    :return: will return a list of the lines of the file, it may be a list of lists depending
    """
    f = open(file_name, 'rt')
    lines = f.readlines()
    f.close()

    def make_float_vec(l, vect=False, split_v=" "):
        if vect:
            rl = l.strip('\n').split(split_v)
            for i in range(len(rl)):
                rl[i] = float(rl[i])
        else:
                rl = float(l)
        return rl

    def make_int_vec(l, vect=False, split_v=" "):
        if vect:
            rl = l.strip('\n').split(split_v)
            for i in range(len(rl)):
                rl[i] = int(rl[i])
        else:
                rl = int(l)
        return rl

    ret_l = list()

    labels = list()

    cnt = 0
    for l in lines:
        if vectorize:
            if convert:
                if label and cnt == 0:
                    labels = l.strip('\n').split(' ')
                else:
                    if convert_type == 'int':
                        ret_l.append(make_int_vec(l, vectorize, split_val))
                    elif convert_type == 'float':
                        ret_l.append(make_float_vec(l, vectorize, split_val))
                cnt += 1
            else:
                rl = l.strip('\n').split(split_val)
                for i in range(len(rl)):
                    ret_l.append(rl)
        else:
            ret_l.append(l)

    if label:
        return ret_l, labels
    else:
        return ret_l


'''
    The below variables are given as standard names
'''
wrk_dir = 'UTK-Peer-Data'
utk_peers_csv = 'UTK-peers.csv'
utk_data_file = 'UTK-peers_data.dt'
imputated_data = 'UTK-peers-data_avg.dt'

utk_peers_json = 'UTK-peers-json.json'

peer_names = 'utk_peer_names.dt'
header_names = 'utk_peer_headers.dt'

attrib_file = 'utk_attribs.dt'

basic_stat_file = 'utk_peer_stats.dt'

ign_list = ['HBC']

extra_ignores = list([0, 1])


def create_needed_files(csv_name=utk_peers_csv, data_file_name=utk_data_file, json_name=utk_peers_json,
                        imputated_data_file=imputated_data, obs_names=peer_names, attrib_names=header_names,
                        ignore_list=ign_list, additional_ignores=[0,1],verbose=True, attrib_file=attrib_file,
                        stat_file=basic_stat_file, dir_name=wrk_dir):
    """This will create a set of needed files for data analysis

    :param csv_name: name of csv file to process
    :param data_file_name: optional, name of data file to create
    :param json_name: optional, name of json file to create
    :param imputated_data_file: optional, name of imputated data file
    :param obs_names: optional, name of the observations file containing the names of each observation
    :param attrib_names: optional, name of the file to be created that will contain the names of the attributes of data
    :param ignore_list: optional, list of names of attribute that should be removed from main list
    :param additional_ignores: optional, list of names of attribute that should be added to ignore list
    :param attrib_file: optional, name you want to write the attribute names to
    :param verbose: optional, used to display some of the data created and loaded
    :param stat_file: optional, name of file to load stats into
    :param dir_name: optional, name of working directory
    :return: None
    """

    os.chdir(dir_name)

    # create a json file containing data from csv file given
    # returns the observation names(s_names), and attribute, names
    s_names, headers, bad_headers = process_csv_to_json(csv_name, json_name)

    if verbose:
        print(headers)

    for i in additional_ignores:
        ignore_list.append(headers[i])

    print('the headers to ignore are:')
    print(ignore_list)

    for header in bad_headers:
        ignore_list = add_unique_element(ignore_list, header)

    print('the headers to ignore are2:')
    print(ignore_list)

    # create two files:
    # one for the observation names
    # one for the attribute labes
    write_list_to_file(s_names, obs_names)
    write_list_to_file(headers, attrib_names)

    # create a basic data array from json file created above
    d_array = process_json_to_data_array(json_name, s_names, headers)

    # remove the items in the ignore list from the attribute labels
    attrib_labels = remove_list(headers, ignore_list)

    if verbose:
        print(attrib_labels)

    print(len(attrib_labels))

    write_list_to_file(attrib_labels, attrib_file)

    write_data_array_to_file(data_file_name, d_array,  attribs=attrib_labels, delimeter=' ', label_delim=' ')

    utk_labels, utk_data, np_utk_data = load_numpy_da_file(data_file_name, labels=True, attrib_delim=' ')

    #print(utk_data.shape)

    # TODO: write this to a file
    attrib_array = get_attrib_array(np_utk_data)

    basic_stats = get_basic_stats(np_utk_data)

    write_list_to_file(basic_stats, stat_file)

    fix_list = np_utk_data.tolist()

    write_data_array_to_file(imputated_data_file, fix_list,  attribs=attrib_labels, delimeter=' ')

    return


def load_data_files(json_file=utk_peers_json, imputated_file=imputated_data, data_file=utk_data_file,
                    obs_names=peer_names, attrib_name=header_names, att_file=attrib_file, stats_file=basic_stat_file,
                    dir_name=wrk_dir):
    # change directory to given one
    os.chdir(dir_name)

    # load the data file
    utk_labels, utk_data, np_utk_data = load_numpy_da_file(data_file, labels=True, attrib_delim=' ')
    s_name = load_list_from_file(obs_names)
    headers = load_list_from_file(attrib_name)
    attribs_l = load_list_from_file(att_file)

    basic_stats = load_list_from_file(stats_file, convert=True, convert_type='float', vectorize=True, split_val=' ')

    imputated_d, s = load_data_from_file(imputated_file)

    ret_list = list()
    ret_list.append(utk_labels)     # 0
    ret_list.append(utk_data)       # 1
    ret_list.append(np_utk_data)    # 2
    ret_list.append(imputated_d)    # 3
    ret_list.append(s_name)         # 4
    ret_list.append(headers)        # 5
    ret_list.append(attribs_l)      # 6
    ret_list.append(basic_stats)    # 7

    return ret_list

