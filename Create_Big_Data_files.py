from BigDataProcessor import*
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

   # os.chdir(wrk_dir)

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
wrk_dir = 'IPEDS-Big-Data'

ipeds_big_csv = 'IPEDS-big-trimmed.csv'
ipeds_big_data = 'IPEDS_Big_data.dt'
ipeds_big_json = 'IPEDS-big-trimmed-json.json'
imputated_data = 'IPEDS-Big-imp.dt'


peer_names = 'IPEDS_peer_names.dt'
header_names = 'IPEDS_peer_headers.dt'

attrib_file = 'IPEDS_attribs.dt'

basic_stat_file = 'IPEDS-Big_stats.dt'

ign_list = ['Institution Name', 'Average net price-students awarded grant or scholarship aid  2015-16 (SFA1516)']

extra_ignores = list([1,2])


def create_needed_files(csv_name=ipeds_big_csv, data_file_name=ipeds_big_data, json_name=ipeds_big_json,
                        imputated_data_file=imputated_data, obs_names=peer_names, attrib_names=header_names,
                        ignore_list=ign_list, additional_ignores=[],verbose=True, attrib_file=attrib_file,
                        stat_file=basic_stat_file):
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
    :return: None
    """

    os.chdir(wrk_dir)

    # create a json file containing data from csv file given
    # returns the observation names(s_names), and attribute, names
    s_names, headers, bad_dict = process_csv_to_json(csv_name, json_name)

    print(headers)
    print(len(headers)*.50)
    print(len(headers))

    print('number od schools')
    print(len(s_names))
    bad_r_c = 0
    bad_l = list()

    if verbose:
        print(len(s_names))
        print(s_names)
        print(len(bad_dict))

    # look through observations and get ready to remove those that
    # have have more than 25% missing data
    # uses bad_dict with is of the form [row_num : num_of_missing_data_points]
    for i in bad_dict:
        if bad_dict[i] > len(headers) * .50:
            bad_l.append(int(i))

    print('There are {:d} observations missing more than 25% of data entries'.format(len(bad_l)))
    print('Which is only {:f} percent of the total'.format(float(float(len(bad_l)/float(len(s_names))))))
    print('So we shall discard them')

        # ignore_list.append(headers[i])

    new_s_names = list()
    for i in range(len(s_names)):
        if i not in bad_l:
            new_s_names.append(s_names[i])

    print('There are now {:d} names left'.format(len(new_s_names)))

    # create two files:
    # one for the observation / school names
    # one for the attribute labels
    write_list_to_file(new_s_names, obs_names)
    write_list_to_file(headers, attrib_names)

    # create a basic data array from json file created above
    #d_array = process_json_to_data_array(json_name, new_s_names, headers, bad_l=bad_l)

    #print('the length of darray is {:d}'.format(len(d_array)))

    # remove the items in the ignore list from the attribute labels
    attrib_labels = remove_list(headers, ignore_list)

    # create a basic data array from json file created above
    d_array = process_json_to_data_array(json_name, new_s_names, attrib_labels, bad_l=bad_l)
    print('the length of darray is {:d}'.format(len(d_array)))

    if verbose:
        print(attrib_labels)

    write_list_to_file(attrib_labels, attrib_file)

    write_data_array_to_file(data_file_name, d_array,  attribs=attrib_labels, delimeter=' ')

    utk_labels, utk_data, np_utk_data = load_numpy_da_file(data_file_name, labels=True, attrib_delim=' ')

    print('length of data')
    print(len(utk_data))
    print('length of np data')
    print(len(np_utk_data))

    # TODO: write this to a file
    attrib_array = get_attrib_array(np_utk_data)

    basic_stats = get_basic_stats(np_utk_data)

    write_list_to_file(basic_stats, stat_file)

    fix_list = np_utk_data.tolist()

    write_data_array_to_file(imputated_data_file, fix_list,  attribs=attrib_labels, delimeter=' ')

    return


def load_data_filesb(json_file=ipeds_big_json, imputated_file=imputated_data, data_file=ipeds_big_data,
                    obs_names=peer_names, attrib_name=header_names, att_file=attrib_file, stats_file=basic_stat_file):

    os.chdir("IPEDS-Big-Data")

    utk_labels, utk_data, np_utk_data = load_numpy_da_file(data_file, labels=True, attrib_delim=' ')
    s_name = load_list_from_file(obs_names)
    headers = load_list_from_file(attrib_name)
    attribs_l = load_list_from_file(att_file)

    basic_stats = load_list_from_file(stats_file, convert=True, convert_type='float', vectorize=True, split_val=' ')

    imputated_d, s = load_data_from_file(imputated_file)

    ret_list = list()
    ret_list.append(utk_labels)
    ret_list.append(utk_data)
    ret_list.append(np_utk_data)
    ret_list.append(imputated_d)
    ret_list.append(s_name)
    ret_list.append(headers)
    ret_list.append(attribs_l)
    ret_list.append(basic_stats)

    return ret_list

