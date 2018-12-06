import json
import csv
import numpy
import matplotlib.pyplot as plt

utk_peers_csv = 'UTK-peers.csv'
utk_data_file = 'UTK-peers_data.dt'

utk_peers_json = 'UTK-peers-json.json'

to_watch = ['Entry_Number', 'Name', 'HBC']
to_fix = [1, 5, 6, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38,
          39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65]


# ---------------------------------------------writing data to files--------------------------------------------------
def write_row_to_file(f, row, delim):

    l = len(row)
    s = l-1

    for i in range(0, l):

        if isinstance(row[i], str):
            f.write(row[i])
        else:
            f.write(str(row[i]))
        if i != s:
            f.write(delim)
        else:
            f.write('\n')
    return


def write_data_array_to_file(in_file, d_array, **kwargs):

    attribs = kwargs.get('attribs', list())
    delim = kwargs.get('delimeter', ' ')
    label_delim = kwargs.get('label_delim', ',')

    f = open(in_file, 'w')

    # write the attribs

    if len(attribs) > 0:
        write_row_to_file(f, attribs, label_delim)

    for row in d_array:
        write_row_to_file(f, row, delim)
# --------------------------------------------------------------------------------------------------------------------


# ------------------------------------------Loading data from files----------------------------------------------------
# used to load the data in the given file
def load_numpy_da_file(f, **kwargs):

    labels = kwargs.get('labels', False)
    data_delim = kwargs.get('data_delim', ' ')
    attrib_delim = kwargs.get('attrib_delim', ',')
    imputation = kwargs.get('imputation', 'average')

    # grab data to load
    d2l = open(f, 'r')

    # create an array of the lines from the file
    d_a = d2l.readlines()

    # close the file
    d2l.close()

    #
    reg_a = list()


    data = list()
    label = list()

    col_l = len(d_a[1].strip('\n').split(data_delim))

    col_list = list(range(0, col_l))
    bd_data = {}

    for r in range(len(d_a)):
        # if there are labels and this is the
        # row for them grab them
        if labels and r == 0:
            label = d_a[r].strip('\n').split(attrib_delim)
        else:
            load_l = list()
            d_row = d_a[r].strip('\n').split(data_delim)
            for c in range(len(d_row)):
                val = d_row[c]

                if val == '#N/A':
                    load_l.append(-99.9)
                    # keep track of the bad data points
                    if c in bd_data:
                        bd_data[c].append(r-1)
                    else:
                        l = list()
                        l.append(r-1)
                        bd_data[c] = l

                else:
                    load_l.append(float(val))
                    if float(val) == -99.9:
                        if c in bd_data:
                            bd_data[c].append(r-1)
                        else:
                            l = list()
                            l.append(r-1)
                            bd_data[c] = l
            # load the row of data
            data.append(load_l)

    raw_data = numpy.array(data, dtype=numpy.float)

    if imputation == 'average':
        data = perform_imputation(list(raw_data), list(data), bd_data)

    raw_data = numpy.array(data, dtype=numpy.float)
    return label, data, raw_data
# ---------------------------------------------------------------------------------------------------------------------


# -------------------------------------Data Manipulation---------------------------------------------------------------
def get_attrib_array(np_data, **kwargs):

    row_col = np_data.shape

    row_s = row_col[0]
    col_s = row_col[1]

    data_trns = numpy.transpose(np_data)

    attrib_array = numpy.array(np_data[:,0])

    return data_trns


def get_basic_stats(np_data_a):
    mu_a = list()
    std_a = list()
    min_a = list()
    max_a = list()
    row_col = np_data_a.shape
    rows = row_col[0]
    cols = row_col[1]
    for i in range(cols):
        mu_a.append(numpy.mean(np_data_a[:,i]))
        std_a.append(numpy.std(np_data_a[:,i]))
        min_a.append(numpy.min(np_data_a[:,i]))
        max_a.append(numpy.max(np_data_a[:,i]))

    ret_list = [mu_a, std_a, min_a, max_a]

    return ret_list


def perform_imputation(data, reg_data,  bad_data, **kwargs):
    imputation = kwargs.get('imputation', 'average')

    dtran = numpy.transpose(data)

    if imputation == 'average':
        #print('Performing Average imputation')
        for col in bad_data:
            rmv_l = bad_data[col]
            #print(dtran[col].tolist())
            avg_v = numpy.mean(numpy.delete(dtran[col], rmv_l))
            #print('The average is {:f} of col {:d}'.format(avg_v, col))
            for val in rmv_l:
                #print('b4',reg_data[val][col])
                reg_data[val][col] = numpy.around(avg_v, 0)
                #print('after',reg_data[val][col])
        return reg_data
    elif imputation == 'discard':
        return data
    elif imputation == 'linear regression':
        return data
# ---------------------------------------------------------------------------------------------------------------------


# -----------------------------------------String manipulation methods-------------------------------------------------
# prepares a numerical string to be converted to a number
def fix_str_to_num(strg):
    #print('fixing {:s}'.format(strg))
    comma_cnt = strg.count(',')

    spce_cnt = strg.count(' ')

    #remove extra space
    for i in range(0, spce_cnt):
        stp = strg.find(' ')
        if stp != -1:
            strg = strg[:stp] + strg[stp + 1:]

    #remove commas
    for i in range(0, comma_cnt):
        stp = strg.find(',')
        if stp != -1:
            strg = strg[:stp] + strg[stp + 1:]

    #remove dollars signs
    strg = strg.strip('$')

    return strg
# --------------------------------------------------------------------------------------------------------------------


# ---------------------------------------csv and json processing methods

def add_unique_element(l, item):
    if item not in l:
        l.append(item)
    return l

def process_csv_to_json(csv_file_name, json_file_name, headers=True, obs_name=True, ign_l=[0,1], name_idx =1):
    csv_file = open(csv_file_name, 'r')
    json_file = open(json_file_name, 'w')
    dreader = csv.DictReader(csv_file)
    headers = dreader.fieldnames

    school_names = list()

    bad_headers = list()

    limit = 56
    cnt = 0
    for row in dreader:
        school_names.append(row['Name'])
        for idx in range(2, len(headers)):
            header = headers[idx]
            val = row[header]

            #if idx in to_fix:
            #    row[header] = fix_str_to_num(val)
            if not val.isnumeric():
                row[header] = fix_str_to_num(val)
                if row[header].isalpha():
                    i = 0
                    bad_headers = add_unique_element(bad_headers, header)

        json.dump(row, json_file)
        json_file.write('\n')
    csv_file.close()
    json_file.close()
    return school_names, headers, bad_headers


def get_data_from_json(json_file_name, s_names, headers, bd_ident=-99.9,
                       stop_colls=('HBC', '2014 Med School', 'Vet School')):
    json_file = open(json_file_name, 'r')
    json_text_array = json_file.readlines()
    json_file.close()
    stop_colls.append(headers[1])
    limit = len(json_text_array)


def process_json_to_data_array(json_file_name, s_names, headers, bd_ident=-99.9,
                               stop_colls=['HBC', '2014 Med School', 'Vet School']):

    json_file = open(json_file_name, 'r')
    json_text_array = json_file.readlines()
    json_file.close()

    # get the number of shools in the file
    number_of_schools = len(json_text_array)

    # remove HBC because all or the same
    #stop_colls = ['HBC', '2014 Med School', 'Vet School']


    #stop_colls.append(headers[0])
    stop_colls.append(headers[1])

    med2014dict = {}
    vetschool = {}

    data_array = list()

    for i in range(0, 57):
        #load the json formatted text into a dictionary called file_text
        file_text = json.loads(json_text_array[i])
        c = 0
        attrib_list = list()
        for idx in range(len(headers)):
            if idx == 0:
                bad_row = headers[idx]
                continue
            #get the header at idx
            entry = headers[idx]
            c += 1
            # grab the data associated with the name entry
            val = file_text[entry]
            val = val.strip('$')

            if entry not in stop_colls:
                if entry == 'Total Faculty':
                    val = float(val)
                elif val.isnumeric():
                    val = float(val)
                else:
                    if val not in s_names and val != '-':
                        try:
                            val = float(val)
                        except ValueError:
                            if val == '':
                                val = bd_ident
                    elif val == '-' or val == '#N/A':
                        val = bd_ident
                attrib_list.append(val)

            else:
                jjk=0
                '''
                if entry == stop_colls[1]:
                    if val in med2014dict:
                        med2014dict[val] += 1
                    else:
                        med2014dict[val] = 1
                    if val == '':
                        attrib_list.append(1)
                    elif val == 'pre clin':
                        attrib_list.append(2)
                    else:
                        attrib_list.append(3)
                elif entry == stop_colls[2]:
                    if val in vetschool:
                        vetschool[val] += 1
                    else:
                        vetschool[val] = 1
                    if val == 'x':
                        attrib_list.append(1)
                    else:
                        attrib_list.append(0)
                '''

        data_array.append(attrib_list)

    num_schools = len(json_text_array)
    return data_array
# ---------------------------------------------------------------------------------------------------------------------

# -------------------------------------Normalization Methods----------------------------------------------------------


def normalize_data_set(x):
    return x - x.min(axis=0) / (x.max(axis=0) - x.min(axis=0))


def z_normalize_data_set(x):
    return x - x.mean(axis=0, dtype=numpy.float64)/x.std(axis=0, dtype=numpy.float64)

# ---------------------------------------------------------------------------------------------------------------------



