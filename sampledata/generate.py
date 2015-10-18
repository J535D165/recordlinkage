# =============================================================================
# generate.py - Auxiliary program to create records using various frequency
#               tables and introduce duplicates with errors
#
# Freely extensible biomedical record linkage (Febrl) Version 0.2.2
# See http://datamining.anu.edu.au/projects/linkage.html
#
# =============================================================================
# AUSTRALIAN NATIONAL UNIVERSITY OPEN SOURCE LICENSE (ANUOS LICENSE)
# VERSION 1.1
#
# The contents of this file are subject to the ANUOS License Version 1.1 (the
# "License"); you may not use this file except in compliance with the License.
# Software distributed under the License is distributed on an "AS IS" basis,
# WITHOUT WARRANTY OF ANY KIND, either express or implied. See the License for
# the specific language governing rights and limitations under the License.
# The Original Software is "generate.py".
# The Initial Developers of the Original Software are Dr Peter Christen
# (Department of Computer Science, Australian National University) and Dr Tim
# Churches (Centre for Epidemiology and Research, New South Wales Department
# of Health). Copyright (C) 2002, 2003 the Australian National University and
# others. All Rights Reserved.
# Contributors:
#
# =============================================================================

"""Module generate.py - Auxiliary program to create records using various
                        frequency tables and introduce duplicates with errors.

   USAGE:
     python generate.py [output_file] [num_originals] [num_duplicates]
                        [max_duplicate_per_record] [distribution]     

   ARGUMENTS:
     output_file               Name of the output file (currently this is a
                               CSV file).
     num_originals             Number of original records to be created.
     num_duplicates            Number of duplicate records to be created.
     max_duplicate_per_record  The maximal number of duplicates that can be
                               created for one original record.
     distribution              The probability distribution used to create
                               the duplicates (i.e the number of duplicates for
                               one original).
                               Possible are: - uniform
                                             - poisson
                                             - zipf

   DESCRIPTION:
     This program can be used to create a data set with records that contain
     randomly created names and addresses (using frequency files), dates and
     identifier numbers. Duplicate records will then be created following a
     given probability distribution, with various errors introduced.

     Various parameters on how theses duplicates are created can be given
     within the program, see below.


   TODO:
     - Fix ZIPF distribution

     - Allow various probability distributions for fields oftype 'date' and
       'iden' (using a new keyword in field dictionaries).

     - Try to find real world error distributions for typographical errors and
       integrate them into the random error creation

     - Add random word spilling between fields (similar to field swapping)
     - Improve performance (loading and creating frequency tables)
"""

# =============================================================================
# Imports go here

import math
import os
import random
import string
import sys
import time
import xreadlines

# Initialise random number generator  - - - - - - - - - - - - - - - - - - - - -
#
random.seed()

# Set the following flag to True for verbose output, otherwise to False - - - -
#
VERBOSE_OUTPUT = True

# =============================================================================
#
# For each field (attribute), a dictionary has to be defined with the following
# keys (probabilities can have values between 0.0 and 1.0):
# - name           The field name to be used when a header is written into the
#                  output file
# - type           The type of the field. Possible are:
#                  'freq' (for fields that use a frequency table with field
#                          values)
#                  'date' (for date fields in a certain range)
#                  'iden' (for numerical identifier fields in a certain range)
# - char_range     The range of random characters that can be introduced. Can
#                  be one of 'alpha', 'digit', or 'alphanum'
# - freq_file      The name of a frequency file (for fields of type 'freq'
#                  only)
# - start_date     A start date (for fields of type 'date' only), must be a
#                  tuple (day,month,year)
# - end_date       A end date (for fields of type 'date' only), must be a
#                  tuple (day,month,year)
# - start_id       A start identification number (for fields of type 'iden'
#                  only)
# - end_id         A end identification number (for fields of type 'iden' only)
#
# - ins_prob       Probability to insert a character into a field
# - del_prob       Probability to delete a character from a field
# - sub_prob       Probability to substitute a character in a field with
#                  another character
# - trans_prob     Probability to transpose two characters in a field
# - val_swap_prob  Probability to swap the value in a field with another
#                  (randomly selected) value for this field (taken from this
#                  field's look-up table)
# - spc_ins_prob   Probability to insert a space into a field (thus splitting
#                  a word)
# - spc_del_prob   Probability to delete a space (if available) in a field (and
#                  thus merging two words)
# - miss_prob      Probability to set a field value to missing (empty)

givenname_dict = {'name':'given_name',
                  'type':'freq',
            'char_range':'alpha',
             'freq_file':'data/givenname.csv',
              'ins_prob':0.03,
              'del_prob':0.04,
              'sub_prob':0.05,
            'trans_prob':0.03,
         'val_swap_prob':0.08,
          'spc_ins_prob':0.01,
          'spc_del_prob':0.00,
             'miss_prob':0.02}

surname_dict = {'name':'surname',
                'type':'freq',
          'char_range':'alpha',
           'freq_file':'data/surname.csv',
              'ins_prob':0.05,
              'del_prob':0.04,
              'sub_prob':0.06,
            'trans_prob':0.05,
         'val_swap_prob':0.05,
          'spc_ins_prob':0.01,
          'spc_del_prob':0.01,
             'miss_prob':0.01}

streetnumber_dict = {'name':'street_number',
                     'type':'freq',
               'char_range':'digit',
                'freq_file':'data/streetnumber.csv',
                 'ins_prob':0.0,
                 'del_prob':0.01,
                 'sub_prob':0.0,
               'trans_prob':0.02,
            'val_swap_prob':0.10,
             'spc_ins_prob':0.0,
             'spc_del_prob':0.0,
                'miss_prob':0.03}

address1_dict = {'name':'address_1',
                 'type':'freq',
           'char_range':'alpha',
            'freq_file':'data/address1.csv',
             'ins_prob':0.05,
             'del_prob':0.05,
             'sub_prob':0.07,
           'trans_prob':0.05,
        'val_swap_prob':0.02,
         'spc_ins_prob':0.05,
         'spc_del_prob':0.05,
            'miss_prob':0.02}

address2_dict = {'name':'address_2',
                 'type':'freq',
           'char_range':'alpha',
            'freq_file':'data/address2.csv',
             'ins_prob':0.04,
             'del_prob':0.04,
             'sub_prob':0.08,
           'trans_prob':0.10,
        'val_swap_prob':0.01,
         'spc_ins_prob':0.10,
         'spc_del_prob':0.05,
            'miss_prob':0.09}

suburb_dict = {'name':'suburb',
               'type':'freq',
           'char_range':'alpha',
          'freq_file':'data/suburb.csv',
           'ins_prob':0.02,
           'del_prob':0.03,
           'sub_prob':0.07,
         'trans_prob':0.05,
      'val_swap_prob':0.05,
       'spc_ins_prob':0.02,
       'spc_del_prob':0.01,
          'miss_prob':0.01}

postcode_dict = {'name':'postcode',
                 'type':'freq',
           'char_range':'digit',
            'freq_file':'data/postcode.csv',
             'ins_prob':0.00,
             'del_prob':0.00,
             'sub_prob':0.05,
           'trans_prob':0.10,
        'val_swap_prob':0.01,
         'spc_ins_prob':0.0,
         'spc_del_prob':0.0,
            'miss_prob':0.0}

state_dict = {'name':'state',
              'type':'freq',
        'char_range':'alpha',
         'freq_file':'data/state.csv',
          'ins_prob':0.0,
          'del_prob':0.0,
          'sub_prob':0.01,
        'trans_prob':0.01,
     'val_swap_prob':0.02,
      'spc_ins_prob':0.0,
      'spc_del_prob':0.0,
         'miss_prob':0.01}

dob_dict = {'name':'date_of_birth',
            'type':'date',
      'char_range':'digit',
      'start_date':(01,01,1900),
        'end_date':(31,12,1999),
        'ins_prob':0.0,
        'del_prob':0.0,
        'sub_prob':0.01,
      'trans_prob':0.01,
   'val_swap_prob':0.04,
    'spc_ins_prob':0.0,
    'spc_del_prob':0.0,
       'miss_prob':0.02}

ssid_dict = {'name':'soc_sec_id',
             'type':'iden',
       'char_range':'digit',
         'start_id':1000000,
           'end_id':9999999,
         'ins_prob':0.0,
         'del_prob':0.0,
         'sub_prob':0.02,
       'trans_prob':0.03,
    'val_swap_prob':0.04,
     'spc_ins_prob':0.0,
     'spc_del_prob':0.0,
        'miss_prob':0.00}

# -----------------------------------------------------------------------------
# Now add all field dictionaries into a list according to how they should be
# saved in the output file

field_list = [givenname_dict, surname_dict, streetnumber_dict, address1_dict,
              address2_dict, suburb_dict, postcode_dict, state_dict,
              dob_dict, ssid_dict]

# -----------------------------------------------------------------------------
# Flag for writing a header line (keys 'name' of field dictionaries)

save_header = True  # Set to 'False' if no header should be written

# -----------------------------------------------------------------------------
# Probabilities (between 0.0 and 1.0) for swapping values between two fields
# Use field names as defined in the field directories (keys 'name').

field_swap_prob = {('address_1', 'address_2'):0.05,
                   ('given_name', 'surname'):0.07}

# -----------------------------------------------------------------------------
# Probabilities (between 0.0 and 1.0) for creating a typographical error (a new
# character) in the same row or the same column. This is used in the random
# selection of a new character in the 'sub_prob' (substitution of a character
# in a field)

single_typo_prob = {'same_row':0.4,
                    'same_col':0.3}

# -----------------------------------------------------------------------------
# String to be inserted for missing values

missing_value = ''

# =============================================================================
# Nothing to be changed below here
# =============================================================================

# =============================================================================
# Functions used by the main program come here

def error_position(input_string, len_offset):
  """A function that randomly calculates an error position within the given
     input string and returns the position as integer number 0 or larger.

     The argument 'len_offset' can be set to an integer (e.g. -1, 0, or 1) and
     will give an offset relative to the string length of the maximal error
     position that can be returned.

     Errors do not likely appear at the beginning of a word, so a gauss random
     distribution is used with the mean being one position behind half the
     string length (and standard deviation 1.0)
  """

  str_len = len(input_string)
  max_return_pos = str_len - 1 + len_offset  # Maximal position to be returned

  if (str_len == 0):
    return None  # Empty input string

  mid_pos = (str_len + len_offset) / 2 + 1

  random_pos = random.gauss(float(mid_pos), 1.0)
  random_pos = max(0,int(round(random_pos)))  # Make it integer and 0 or larger

  return min(random_pos, max_return_pos)

# -----------------------------------------------------------------------------

def error_character(input_char, char_range):
  """A function which returns a character created randomly. It uses row and
     column keyboard dictionaires.
  """

  # Keyboard substitutions gives two dictionaries with the neigbouring keys for
  # all letters both for rows and columns (based on ideas implemented by
  # Mauricio A. Hernandez in his dbgen).
  #
  rows = {'a':'s',  'b':'vn', 'c':'xv', 'd':'sf', 'e':'wr', 'f':'dg', 'g':'fh',
          'h':'gj', 'i':'uo', 'j':'hk', 'k':'jl', 'l':'k',  'm':'n',  'n':'bm',
          'o':'ip', 'p':'o',  'q':'w',  'r':'et', 's':'ad', 't':'ry', 'u':'yi',
          'v':'cb', 'w':'qe', 'x':'zc', 'y':'tu', 'z':'x',
          '1':'2',  '2':'13', '3':'24', '4':'35', '5':'46', '6':'57', '7':'68',
          '8':'79', '9':'80', '0':'9'}

  cols = {'a':'qz', 'b':'g',  'c':'d',  'd':'ec', 'e':'d',  'f':'rv', 'g':'tb',
          'h':'yn', 'i':'k',  'j':'um', 'k':'i',  'l':'o',  'm':'j',  'n':'h',
          'o':'l',  'p':'p',  'q':'a',  'r':'f',  's':'wx', 't':'g',  'u':'j',
          'v':'f',  'w':'s',  'x':'s',  'y':'h',  'z':'a'}

  rand_num = random.random()  # Create a random number between 0 and 1

  if (char_range == 'digit'):

    # A randomly chosen neigbouring key in the same keyboard row
    #
    if (input_char.isdigit()) and (rand_num <= single_typo_prob['same_row']):
      output_char = random.choice(rows[input_char])
    else:
      choice_str =  string.replace(string.digits, input_char, '')
      output_char = random.choice(choice_str)  # A randomly choosen digit

  elif (char_range == 'alpha'):

    # A randomly chosen neigbouring key in the same keyboard row
    #
    if (input_char.isalpha()) and (rand_num <= single_typo_prob['same_row']):
      output_char = random.choice(rows[input_char])

    # A randomly chosen neigbouring key in the same keyboard column
    #
    elif (input_char.isalpha()) and \
       (rand_num <= (single_typo_prob['same_row'] + \
                     single_typo_prob['same_col'])):
      output_char = random.choice(cols[input_char])
    else:
      choice_str =  string.replace(string.lowercase, input_char, '')
      output_char = random.choice(choice_str)  # A randomly choosen letter

  else:  # Both letters and digits possible

    # A randomly chosen neigbouring key in the same keyboard row
    #
    if (rand_num <= single_typo_prob['same_row']):
      if (input_char in rows):
        output_char = random.choice(rows[input_char])
      else:
        choice_str =  string.replace(string.lowercase+string.digits, \
                                     input_char, '')
        output_char = random.choice(choice_str)  # A randomly choosen character

    # A randomly chosen neigbouring key in the same keyboard column
    #
    elif (rand_num <= (single_typo_prob['same_row'] + \
                       single_typo_prob['same_col'])):
      if (input_char in cols):
        output_char = random.choice(cols[input_char])
      else:
        choice_str =  string.replace(string.lowercase+string.digits, \
                                     input_char, '')
        output_char = random.choice(choice_str)  # A randomly choosen character

    else:
      choice_str =  string.replace(string.lowercase+string.digits, \
                                   input_char, '')
      output_char = random.choice(choice_str)  # A randomly choosen character

  return output_char

# -----------------------------------------------------------------------------

# Some simple funcions used for date conversions follow
# (based on functions from the 'normalDate.py' module by Jeff Bauer, see:
# http://starship.python.net/crew/jbauer/normalDate/)

days_in_month = [[31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], \
                 [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]]

def first_day_of_year(year):
  """Calculate the day number (relative to 1 january 1900) of the first day in
     the given year.
  """

  if (year == 0):
    print 'Error: A year value of 0 is not possible'
    raise Exception

  elif (year < 0):
    first_day = (year * 365) + int((year - 1) / 4) - 693596
  else:  # Positive year
    leap_adj = int ((year + 3) / 4)
    if (year > 1600):
      leap_adj = leap_adj - int((year + 99 - 1600) / 100) + \
                 int((year + 399 - 1600) / 400)

    first_day = year * 365 + leap_adj - 693963

    if (year > 1582):
      first_day -= 10

  return first_day

# -----------------------------------------------------------------------------

def is_leap_year(year):
  """Determine if the given year is a leap year. Returns 0 (no) or 1 (yes).
  """

  if (year < 1600):
    if ((year % 4) != 0):
      return 0
    else:
      return 1

  elif ((year % 4) != 0):
    return 0

  elif ((year % 100) != 0):
    return 1

  elif ((year % 400) != 0):
    return 0

  else:
    return 1

# -----------------------------------------------------------------------------

def epoch_to_date(daynum):
  """Convert an epoch day number into a date [day, month, year], with
     day, month and year being strings of length 2, 2, and 4, respectively.
     (based on a function from the 'normalDate.py' module by Jeff Bauer, see:
     http://starship.python.net/crew/jbauer/normalDate/)

  USAGE:
    [year, month, day] = epoch_to_date(daynum)

  ARGUMENTS:
    daynum  A integer giving the epoch day (0 = 1 January 1900)

  DESCRIPTION:
    Function for converting a number of days (integer value) since epoch time
    1 January 1900 (integer value) into a date tuple [day, month, year].

  EXAMPLES:
    [day, month, year] = epoch_to_date(0)      # returns ['01','01','1900']
    [day, month, year] = epoch_to_date(37734)  # returns ['25','04','2003']
  """

  if (not (isinstance(daynum, int) or isinstance(daynum, long))):
    print 'Error: Input value for "daynum" is not of integer type: %s' % \
          (str(daynum))
    raise Exception

  if (daynum >= -115860):
    year = 1600 + int(math.floor((daynum + 109573) / 365.2425))
  elif (daynum >= -693597):
    year = 4 + int(math.floor((daynum + 692502) / 365.2425))
  else:
    year = -4 + int(math.floor((daynum+695058) / 365.2425))

  days = daynum - first_day_of_year(year) + 1

  if (days <= 0):
    year -= 1  
    days = daynum - first_day_of_year(year) + 1

  days_in_year = 365 + is_leap_year(year)  # Adjust for a leap year

  if (days > days_in_year):
    year += 1
    days = daynum - first_day_of_year(year) + 1

  # Add 10 days for dates between 15 October 1582 and 31 December 1582
  #
  if (daynum >= -115860) and (daynum <= -115783):
    days += 10

  day_count = 0
  month = 12
  leap_year_flag = is_leap_year(year)

  for m in range(12):
    day_count += days_in_month[leap_year_flag][m]
    if (day_count >= days):
      month = m + 1
      break

  # Add up the days in the prior months
  #
  prior_month_days = 0
  for m in range(month-1):
    prior_month_days += days_in_month[leap_year_flag][m]

  day = days - prior_month_days

  day_str =   string.zfill(str(day),2)  # Add '0' if necessary
  month_str = string.zfill(str(month),2)  # Add '0' if necessary
  year_str =  str(year)  # Is always four digits long

  return [day_str, month_str, year_str]

# -----------------------------------------------------------------------------

def date_to_epoch(day, month, year):
  """ Convert a date [day, month, year] into an epoch day number.
     (based on a function from the 'normalDate.py' module by Jeff Bauer, see:
     http://starship.python.net/crew/jbauer/normalDate/)

  USAGE:
    daynum = date_to_epoch(year, month, day)

  ARGUMENTS:
    day    Day value (string or integer number)
    month  Month value (string or integer number)
    year   Year value (string or integer number)

  DESCRIPTION:
    Function for converting a date into a epoch day number (integer value)
    since 1 january 1900.

  EXAMPLES:
    day = date_to_epoch('01', '01', '1900')  # returns 0
    day = date_to_epoch('25', '04', '2003')  # returns 37734
  """

  # Convert into integer values
  #
  try:
    day_int = int(day)
  except:
    print 'Error: "day" value is not an integer'
    raise Exception
  try:
    month_int = int(month)
  except:
    print 'Error: "month" value is not an integer'
    raise Exception
  try:
    year_int = int(year)
  except:
    print 'Error: "year" value is not an integer'
    raise Exception

  # Test if values are within range
  #
  if (year_int <= 1000):
    print 'Error: Input value for "year" is not a positive integer ' + \
          'number: %i' % (year)
    raise Exception

  leap_year_flag = is_leap_year(year_int)

  if (month_int <= 0) or (month_int > 12):
    print 'Error: Input value for "month" is not a possible day number: %i' % \
          (month)
    raise Exception

  if (day_int <= 0) or (day_int > days_in_month[leap_year_flag][month_int-1]):
    print 'Error: Input value for "day" is not a possible day number: %i' % \
          (day)
    raise Exception

  days = first_day_of_year(year_int) + day_int - 1

  for m in range(month_int-1):
    days += days_in_month[leap_year_flag][m]

  if (year_int == 1582):
    if (month_int > 10) or ((month_int == 10) and (day_int > 4)):
      days -= 10

  return days

# =============================================================================
# Start main program

if (len(sys.argv) != 6):
  print 'Five arguments needed with %s:' % (sys.argv[0])
  print '  - Output file name'
  print '  - Number of original records'
  print '  - Number of duplicate records'
  print '  - Maximal number of duplicate records for one original record'
  print '  - Probability distribution for duplicates (uniform, poisson, zipf)'
  print 'All other parameters have to be set within the code'
  sys.exit()

output_file =       sys.argv[1]
num_org_records =   int(sys.argv[2])
num_dup_records =   int(sys.argv[3])
max_num_dups =      int(sys.argv[4])
prob_distribution = sys.argv[5]

if (num_org_records <= 0):
  print 'Error: Number of original records must be positive'
  sys.exit()

if (num_dup_records < 0):
  print 'Error: Number of duplicate records must be zero or positive'
  sys.exit()

if (max_num_dups <= 0):
  print 'Error: Maximal number of duplicate per record must be positive'
  sys.exit()

if (prob_distribution not in ['uniform', 'poisson', 'zipf']):
  print 'Error: Illegal probability distribution: %s' % (prob_distribution)
  print '       Must be one of: "uniform", "poisson", or "zipf"'
  sys.exit()

# Check all user options for validity - - - - - - - - - - - - - - - - - - - - -
#
field_names = []  # Make a list of all field names

# Check if all defined field dictionaries have the necessary keys
#
for i in range(len(field_list)):
  field_dict = field_list[i]

  if (field_dict.get('name','') == ''):
    print 'Error: No field name given for field dictionary'
    raise Exception
  elif (field_dict['name'] == 'rec_id'):
    print 'Error: Illegal field name "rec_id" (used for record identifier)'
    raise Exception
  else:
    field_names.append(field_dict['name'])

  if (field_dict.get('type','') not in ['freq','date','iden']):
    print 'Error: Illegal or no field type given for field "%s": %s' % \
          (field_dict['name'], field_dict.get('type',''))
    raise Exception

  if (field_dict.get('char_range','') not in ['alpha', 'alphanum','digit']):
    print 'Error: Illegal or no random character range given for ' + \
          'field "%s": %s' % (field_dict['name'], \
                              field_dict.get('char_range',''))
    raise Exception

  if (field_dict['type'] == 'freq'):
    if (not field_dict.has_key('freq_file')):
      print 'Error: Field of type "freq" has no file name given'
      raise Exception

  elif (field_dict['type'] == 'date'):
    if (not (field_dict.has_key('start_date') or \
             field_dict.has_key('end_date'))):
      print 'Error: Field of type "date" has no start and/or end date given'
      raise Exception

    else:  # Process start and end date
      start_date = field_dict['start_date']
      end_date =   field_dict['end_date']

      start_epoch = date_to_epoch(start_date[0], start_date[1], start_date[2])
      end_epoch =   date_to_epoch(end_date[0], end_date[1], end_date[2])
      field_dict['start_epoch'] = start_epoch
      field_dict['end_epoch'] =   end_epoch
      field_list[i] = field_dict

  elif (field_dict['type'] == 'iden'):
    if (not (field_dict.has_key('start_id') or \
             field_dict.has_key('end_id'))):
      print 'Error: Field of type "iden" has no start and/or end ' + \
            'identification number given'
      raise Exception

  if (field_dict.get('ins_prob','') == ''):
    print 'Error: No insert probability given in dictionary for field "%s"' % \
          (field_dict['name'])
    raise Exception
  elif (field_dict['ins_prob'] < 0.0) or (field_dict['ins_prob'] > 1.0):
    print 'Error: Illegal value for insert probability in dictionary for ' + \
          'field "%s": %f' % (field_dict['name'], field_dict['ins_prob'])

  if (field_dict.get('del_prob','') == ''):
    print 'Error: No deletion probability given in dictionary for ' + \
          'field "%s"' % (field_dict['name'])
    raise Exception
  elif (field_dict['del_prob'] < 0.0) or (field_dict['del_prob'] > 1.0):
    print 'Error: Illegal value for deletion probability in dictionary for' + \
          ' field "%s": %f' % (field_dict['name'], field_dict['del_prob'])

  if (field_dict.get('sub_prob','') == ''):
    print 'Error: No substitution probability given in dictionary for '+ \
          'field "%s"' % (field_dict['name'])
    raise Exception
  elif (field_dict['sub_prob'] < 0.0) or (field_dict['sub_prob'] > 1.0):
    print 'Error: Illegal value for substitution probability in dictionary' + \
          ' for field "%s": %f' % (field_dict['name'], field_dict['sub_prob'])

  if (field_dict.get('trans_prob','') == ''):
    print 'Error: No transposition probability given in dictionary for '+ \
          'field "%s"' % (field_dict['name'])
    raise Exception
  elif (field_dict['trans_prob'] < 0.0) or (field_dict['trans_prob'] > 1.0):
    print 'Error: Illegal value for transposition probability in dictionary' +\
          ' for field "%s": %f' % (field_dict['name'], \
                                   field_dict['trans_prob'])

  if (field_dict.get('val_swap_prob','') == ''):
    print 'Error: No value swapping probability given in dictionary for '+ \
          'field "%s"' % (field_dict['name'])
    raise Exception
  elif (field_dict['val_swap_prob'] < 0.0) or \
       (field_dict['val_swap_prob'] > 1.0):
    print 'Error: Illegal value for value swapping probability in ' + \
          'dictinary for field "%s": %f' % (field_dict['name'], \
                                            field_dict['val_swap_prob'])

  if (field_dict.get('spc_ins_prob','') == ''):
    print 'Error: No space insertion probability given in dictionary for '+ \
          'field "%s"' % (field_dict['name'])
    raise Exception
  elif (field_dict['spc_ins_prob'] < 0.0) or \
       (field_dict['spc_ins_prob'] > 1.0):
    print 'Error: Illegal value for space insertion probability in ' + \
          'dictionary for field "%s": %f' % (field_dict['name'], \
                                   field_dict['spc_ins_prob'])

  if (field_dict.get('spc_del_prob','') == ''):
    print 'Error: No space deletion probability given in dictionary for '+ \
          'field "%s"' % (field_dict['name'])
    raise Exception
  elif (field_dict['spc_del_prob'] < 0.0) or \
       (field_dict['spc_del_prob'] > 1.0):
    print 'Error: Illegal value for space deletion probability in ' + \
          'dictionary for field "%s": %f' % (field_dict['name'], \
                                   field_dict['spc_del_prob'])

  if (field_dict.get('miss_prob','') == ''):
    print 'Error: No missing value probability given in dictionary for '+ \
          'field "%s"' % (field_dict['name'])
    raise Exception
  elif (field_dict['miss_prob'] < 0.0) or (field_dict['miss_prob'] > 1.0):
    print 'Error: Illegal value for missing value probability in ' + \
          'dictionary for field "%s": %f' % (field_dict['name'], \
                                   field_dict['miss_prob'])

# Create a distribution of the duplicates - - - - - - - - - - - - - - - - - - -
#
prob_dist_list = [0.0]

if (prob_distribution == 'uniform'):  # Uniform distribution of duplicates

  uniform_val = 1.0 / float(max_num_dups)

  for i in range(max_num_dups-1):
    prob_dist_list.append(uniform_val+prob_dist_list[-1])

elif (prob_distribution == 'poisson'):  # Poisson distribution of duplicates

  def fac(n):  # Factorial of an integer number (recursive calculation)
    if (n > 1.0):
      return n*fac(n - 1.0)
    else:
      return 1.0

  poisson_num = []  # A list of poisson numbers
  poisson_sum = 0.0  # The sum of all poisson number

  # The mean (lambda) for the poisson numbers
  #
  mean = 1.0 + (float(num_dup_records) / float(num_org_records))

  for i in range(max_num_dups):
    poisson_num.append((math.exp(-mean) * (mean ** i)) / fac(i))
    poisson_sum += poisson_num[-1]

  for i in range(max_num_dups):  # Scale so they sum up to 1.0
    poisson_num[i] = poisson_num[i] / poisson_sum

  for i in range(max_num_dups-1):
    prob_dist_list.append(poisson_num[i]+prob_dist_list[-1])

elif (prob_distribution == 'zipf'):  # Zipf distribution of duplicates
  zipf_theta = 0.5

  denom = 0.0
  for i in range(num_org_records):
    denom += (1.0 / (i+1) ** (1.0 - zipf_theta))

  zipf_c = 1.0 / denom
  zipf_num = []  # A list of Zipf numbers
  zipf_sum = 0.0  # The sum of all Zipf number

  for i in range(max_num_dups):
    zipf_num.append(zipf_c / ((i+1) ** (1.0 - zipf_theta)))
    zipf_sum += zipf_num[-1]

  for i in range(max_num_dups):  # Scale so they sum up to 1.0
    zipf_num[i] = zipf_num[i] / zipf_sum

  for i in range(max_num_dups-1):
    prob_dist_list.append(zipf_num[i]+prob_dist_list[-1])

print
print 'Create %i original and %i duplicate records' % \
      (num_org_records, num_dup_records)
print '  Distribution of number of duplicates (maximal %i duplicates):' % \
      (max_num_dups)
print '  %s' % (prob_dist_list)

# Load frequency files - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#
print
print 'Step 1: Load and process frequency tables'

freq_files = {}
freq_files_length = {}

for field_dict in field_list:
  field_name = field_dict['name']

  if (field_dict['type'] == 'freq'):  # Check for 'freq' field type

    file_name = field_dict['freq_file']  # Get the corresponding file name

    if (file_name != None):
      try:
        fin = open(file_name)  # Open file for reading
      except:
        print '  Error: Can not open frequency file %s' % (file_name)
        raise Exception
      value_list = []  # List with all values of the frequency file

      for line in xreadlines.xreadlines(fin):
        line = line.strip()
        line_list = line.split(',')
        if (len(line_list) != 2):
          print '  Error: Illegal format in  frequency file %s: %s' % \
                (file_name, line)
          raise Exception

        line_val =  line_list[0].strip()
        line_freq = int(line_list[1])

        # Append value as many times as given in frequency file
        #
        value_list += [line_val]* line_freq

      random.shuffle(value_list)  # Randomly shuffle the list of values

      freq_files[field_name] = value_list
      freq_files_length[field_name] = len(value_list)

    else:
      print '  Error: No file name defined for frequency field "%s"' % \
            (field_dict['name'])
      raise Exception

# Create original records - - - - - - - - - - - - - - - - - - - - - - - - - - -
#
print
print 'Step 2: Create original records'

org_rec = {}  # Dictionary for original records
all_rec = {}  # Dictionary for all records
rec_cnt = 0

while (rec_cnt < num_org_records):
  rec_id = 'rec-%i-org' % (rec_cnt)  # The records identifier

  rec_dict = {'rec_id':rec_id}  # Save record identifier

  # Now randomly create all the fields in a record  - - - - - - - - - - - - - -
  #
  for field_dict in field_list:
    field_name = field_dict['name']

    # Randomly set field values to missing
    #
    if (random.random() <= field_dict['miss_prob']):
      rand_val = missing_value

    elif (field_dict['type'] == 'freq'):  # A frequency file based field
      rand_num = random.randint(0, freq_files_length[field_name]-1)
      rand_val = freq_files[field_name][rand_num]

    elif (field_dict['type'] == 'date'):  # A date field
      rand_num = random.randint(field_dict['start_epoch'], \
                                field_dict['end_epoch']-1)
      rand_date = epoch_to_date(rand_num)
      rand_val = rand_date[2]+rand_date[1]+rand_date[0]  # ISO format: yyyymmdd

    elif (field_dict['type'] == 'iden'):  # A identification number field
      rand_num = random.randint(field_dict['start_id'], \
                                field_dict['end_id']-1)
      rand_val = str(rand_num)

    if (rand_val != missing_value):  # Don't save missing values
      rec_dict[field_name] = rand_val

  rec_data = rec_dict.copy()  # Make a copy of the record dictionary
  del(rec_data['rec_id'])  # Remove the record identifier
  rec_list = rec_data.items()
  rec_list.sort()
  rec_str = str(rec_list)

  if (not all_rec.has_key(rec_str)):  # Check if same record already created
    all_rec[rec_str] = rec_id
    org_rec[rec_id] = rec_dict  # Insert into original records
    rec_cnt += 1

    # Print original record - - - - - - - - - - - - - - - - - - - - - - - - - -
    #
    if (VERBOSE_OUTPUT == True):
      print '  Original:'
      print '    Record ID         : %-30s' % (rec_dict['rec_id'])
      for field_name in field_names:
        print '    %-18s: %-30s' % (field_name, \
                                    rec_dict.get(field_name, missing_value))
      print

  else:
    if (VERBOSE_OUTPUT == True):
      print '***** Record "%s" already crated' % (rec_str)

# Create duplicate records  - - - - - - - - - - - - - - - - - - - - - - - - - -
#
print
print 'Step 2: Create duplicate records'

dup_rec = {}  # Dictionary for original records

rec_cnt = 0
org_rec_used = {}  # Dictionary with record IDs of original records used to
                   # create duplicates

while (rec_cnt < num_dup_records):

  # First determine how many duplicates to create for a record
  #
  rand_num = random.random()  # Random number between 0.0 and 1.0

  num_dups = max_num_dups
  while (prob_dist_list[num_dups-1] > rand_num):
    num_dups -= 1

  # Now find an original record that has so far not been used to create
  # duplicates
  #
  rand_rec_num = random.randint(0, num_org_records)
  org_rec_id = 'rec-%i-org' % (rand_rec_num)

  while (org_rec_used.has_key(org_rec_id) or \
         (not org_rec.has_key(org_rec_id))):
    rand_rec_num = random.randint(0, num_org_records)  # Get new record number
    org_rec_id = 'rec-%i-org' % (rand_rec_num)

  if (VERBOSE_OUTPUT == True):
    print '  Use record %s to create %i duplicates' % (org_rec_id, num_dups)

  d = 0  # Loop counter for duplicates for this record

  # Loop to create duplicate records
  #
  while (d < num_dups) and (rec_cnt < num_dup_records):

    # Duplicate record identifier
    #
    dup_rec_id = 'rec-%i-dup-%i' % (rand_rec_num, d)

    rec_dict = {'rec_id':dup_rec_id}  # Save record identifier

    org_record = org_rec[org_rec_id]  # Get the original record

    # Now randomly modify all the fields in a record  - - - - - - - - - - - - -
    #
    for field_dict in field_list:
      field_name = field_dict['name']

      if (field_dict['char_range'] == 'digit'):
        field_range = string.digits
      elif (field_dict['char_range'] == 'alpha'):
        field_range = string.lowercase
      elif (field_dict['char_range'] == 'alphanum'):
        field_range = string.digits+string.lowercase

      org_field_val = org_record.get(field_name, None) # Get the original value

      dup_field_val = org_field_val  # Make a copy for the duplicate record

      # If the field value is empty (missing), randomly create a new value  - -
      #
      if (dup_field_val == None):

        if (random.random() > field_dict['miss_prob']):
          dup_field_val = missing_value  # Leave it as missing value

        else:  # Create a new value

          if (field_dict['type'] == 'freq'):  # A frequency file based field
            rand_num = random.randint(0, freq_files_length[field_name]-1)
            dup_field_val = freq_files[field_name][rand_num]

          elif (field_dict['type'] == 'date'):  # A date field
            rand_num = random.randint(field_dict['start_epoch'], \
                                      field_dict['end_epoch']-1)
            rand_date = epoch_to_date(rand_num)
            dup_field_val = rand_date[2]+rand_date[1]+rand_date[0] # ISO format

          elif (field_dict['type'] == 'iden'):  # A identification number field
            rand_num = random.randint(field_dict['start_id'], \
                                      field_dict['end_id']-1)
            dup_field_val = str(rand_num)

          if (VERBOSE_OUTPUT == True):
            print '    Exchanged missing value "%s" in field "%s" with: "%s"' \
                  % (missing_value, field_name, dup_field_val)

      # Random exchange of a field value with another value - - - - - - - - - -
      #
      if (random.random() <= field_dict['val_swap_prob']):
        old_field_val = dup_field_val  # Save old value

        if (field_dict['type'] == 'freq'):  # A frequency file based field
          rand_num = random.randint(0, freq_files_length[field_name]-1)
          dup_field_val = freq_files[field_name][rand_num]

        elif (field_dict['type'] == 'date'):  # A date field
          rand_num = random.randint(field_dict['start_epoch'], \
                                    field_dict['end_epoch']-1)
          rand_date = epoch_to_date(rand_num)
          dup_field_val = rand_date[2]+rand_date[1]+rand_date[0]  # ISO format

        elif (field_dict['type'] == 'iden'):  # A identification number field
          rand_num = random.randint(field_dict['start_id'], \
                                    field_dict['end_id']-1)
          dup_field_val = str(rand_num)

        if (VERBOSE_OUTPUT == True):
          print '    Exchanged value in field "%s": "%s" -> "%s"' % \
                    (field_name, old_field_val, dup_field_val)

      # Random substitution of a character  - - - - - - - - - - - - - - - - - -
      #
      if (random.random() <= field_dict['sub_prob']):
        old_field_val = dup_field_val  # Save old value

        # Get an substitution position randomly
        #
        rand_sub_pos = error_position(dup_field_val, 0)

        if (rand_sub_pos != None):  # If a valid position was returned

          old_char = dup_field_val[rand_sub_pos]
          new_char = error_character(old_char, field_dict['char_range'])

          dup_field_val = dup_field_val[:rand_sub_pos] + new_char + \
                          dup_field_val[rand_sub_pos+1:]

          if (VERBOSE_OUTPUT == True):
            print '    Substituted character "%s" with "%s" in field "%s":' % \
                  (old_char, new_char, field_name) + ' "%s" -> "%s"' % \
                  (old_field_val, dup_field_val)

      # Random insertion of a character - - - - - - - - - - - - - - - - - - - -
      #
      if (random.random() <= field_dict['ins_prob']):
        old_field_val = dup_field_val  # Save old value

        # Get an insert position randomly
        #
        rand_ins_pos = error_position(dup_field_val, +1)
        rand_char =    random.choice(field_range)

        if (rand_ins_pos != None):  # If a valid position was returned
          dup_field_val = dup_field_val[:rand_ins_pos] + rand_char + \
                          dup_field_val[rand_ins_pos:]

          if (VERBOSE_OUTPUT == True):
            print '    Inserted character "%s" into field "%s": "%s" -> "%s"' \
                  % (rand_char, field_name, old_field_val, dup_field_val)

      # Random deletion of a character  - - - - - - - - - - - - - - - - - - - -
      #
      if (random.random() <= field_dict['del_prob']) and \
         (len(dup_field_val) > 1):  # Field must have at least 2 characters
        old_field_val = dup_field_val  # Save old value

        # Get a delete position randomly
        #
        rand_del_pos = error_position(dup_field_val, 0)

        del_char = dup_field_val[rand_del_pos]

        dup_field_val = dup_field_val[:rand_del_pos] + \
                        dup_field_val[rand_del_pos+1:]

        if (VERBOSE_OUTPUT == True):
          print '    Deleted character "%s" in field "%s": "%s" -> "%s"' % \
                (del_char, field_name, old_field_val, dup_field_val)

      # Random transposition of two characters  - - - - - - - - - - - - - - - -
      #
      if (random.random() <= field_dict['trans_prob']) and \
         (len(dup_field_val) > 1):  # Field must have at least 2 characters
        old_field_val = dup_field_val  # Save old value

        # Get a transposition position randomly
        #
        rand_trans_pos = error_position(dup_field_val, -1)

        trans_chars = dup_field_val[rand_trans_pos:rand_trans_pos+2]
        trans_chars2 = trans_chars[1] + trans_chars[0]  # Do transposition

        dup_field_val = dup_field_val[:rand_trans_pos] + trans_chars2 + \
                        dup_field_val[rand_trans_pos+2:]

        if (VERBOSE_OUTPUT == True):
          print '    Transposed characters "%s" in field "%s": "%s" -> "%s"' \
                % (trans_chars, field_name, old_field_val, dup_field_val)

      # Random insertion of a space (thus splitting a word) - - - - - - - - - -
      #
      if (random.random() <= field_dict['spc_ins_prob']) and \
         (len(dup_field_val) > 1):  # Field must have at least 2 characters
        old_field_val = dup_field_val  # Save old value

        # Randomly select the place where to insert a space (make sure no
        # spaces are next to this place)
        #
        rand_ins_pos = error_position(dup_field_val, 0)
        while (dup_field_val[rand_ins_pos-1] == ' ') or \
              (dup_field_val[rand_ins_pos] == ' '):
          rand_ins_pos = error_position(dup_field_val, 0)

        dup_field_val = dup_field_val[:rand_ins_pos] + ' ' + \
                        dup_field_val[rand_ins_pos:]

        if (VERBOSE_OUTPUT == True):
          print '    Inserted space " " into field "%s": "%s" -> "%s"' % \
                (field_name, old_field_val, dup_field_val)

      # Random deletion of a space (thus merging two words) - - - - - - - - - -
      #
      if (random.random() <= field_dict['spc_del_prob']) and \
         (' ' in dup_field_val):  # Field must contain a space character
        old_field_val = dup_field_val  # Save old value

        # Count number of spaces and randomly select one to be deleted
        #
        num_spaces = dup_field_val.count(' ')

        if (num_spaces == 1):
          space_ind = dup_field_val.index(' ')  # Get index of the only space
        else:
          rand_space = random.randint(1, num_spaces-1)
          space_ind = dup_field_val.index(' ', 0)  # Get index of first space
          for i in range(rand_space):
            # Get index of following spaces
            space_ind = dup_field_val.index(' ', space_ind)  

        dup_field_val = dup_field_val[:space_ind]+dup_field_val[space_ind+1:]

        if (VERBOSE_OUTPUT == True):
          print '    Deleted space " " from field "%s": "%s" -> "%s"' % \
                    (field_name, old_field_val, dup_field_val)

      # Random set to missing field - - - - - - - - - - - - - - - - - - - - - -
      #
      if (random.random() <= field_dict['miss_prob']):
        old_field_val = dup_field_val  # Save old value

        dup_field_val = missing_value  # Set to a missing value

        if (VERBOSE_OUTPUT == True):
          print '    Set field "%s" to missing value: "%s" -> "%s"' % \
                    (field_name, old_field_val, dup_field_val)

      if (dup_field_val != missing_value):  # Don't save missing values
        rec_dict[field_name] = dup_field_val

    # Random field swapping - - - - - - - - - - - - - - - - - - - - - - - - - -
    #
    for field_pair in field_swap_prob:

      if (random.random() <= field_swap_prob[field_pair]):
        field_name_a = field_pair[0]
        field_name_b = field_pair[1]

        # Make sure both fields are in the record dictionary
        #
        if (rec_dict.has_key(field_name_a) and rec_dict.has_key(field_name_b)):
          field_value_a = rec_dict[field_name_a]
          field_value_b = rec_dict[field_name_b]

          rec_dict[field_name_a] = field_value_b  # Swap field values
          rec_dict[field_name_b] = field_value_a

          if (VERBOSE_OUTPUT == True):
            print '    Swapped fields "%s" and "%s": "%s" <-> "%s"' % \
                  (field_name_a, field_name_b, field_value_a, field_value_b)

    # Now check if the duplicate record differs from the original - - - - - - -
    #
    rec_data = rec_dict.copy()  # Make a copy of the record dictionary
    del(rec_data['rec_id'])  # Remove the record identifier
    rec_list = rec_data.items()
    rec_list.sort()
    rec_str = str(rec_list)

    if (not all_rec.has_key(rec_str)):  # Check if same record already exists
      all_rec[rec_str] = dup_rec_id
      org_rec_used[org_rec_id] = 1

      dup_rec[dup_rec_id] = rec_dict  # Insert into duplicate records
      rec_cnt += 1

      d += 1  # Duplicate counter (loop counter)

      # Print original and duplicate records field by field - - - - - - - - - -
      #
      if (VERBOSE_OUTPUT == True):
        print '  Original and duplicate records:'
        print '    Record ID         : %-30s | %-30s' % \
              (org_record['rec_id'], rec_dict['rec_id'])
        for field_name in field_names:
          print '    %-18s: %-30s | %-30s' % \
                (field_name, org_record.get(field_name, missing_value), \
                 rec_dict.get(field_name, missing_value))
        print

    else:
      if (VERBOSE_OUTPUT == True):
        print '  No random modifications for record "%s" -> Choose another' % \
              (dup_rec_id)

    if (VERBOSE_OUTPUT == True):
      print

# Write output csv file - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#
print
print 'Step 3: Write output file'

all_rec = org_rec  # Merge original and duplicate records
all_rec.update(dup_rec)

# Get all record IDs and shuffle them randomly
#
all_rec_ids = all_rec.keys()
random.shuffle(all_rec_ids)

# Make a list of field names and sort them according to column number
#

field_name_list = ['rec_id']+field_names

# Open output file
#
try:
  f_out = open(output_file,'w')
except:
  print 'Error: Can not write to output file "%s"' % (output_file)
  sys.exit()

# Write header line
#
if (save_header == True):
  header_line = ''
  for field_name in field_name_list:
    header_line = header_line + field_name+ ', '
  header_line = header_line[:-2]
  f_out.write(header_line+os.linesep)

# Loop over all record IDs
#
for rec_id in all_rec_ids:
  rec_dict = all_rec[rec_id]
  out_line = ''
  for field_name in field_name_list:
    out_line = out_line + rec_dict.get(field_name, missing_value) + ', '

  # Remove last comma and space and add line separator
  #
  out_line = out_line[:-2]
  # print out_line
  f_out.write(out_line+os.linesep)

f_out.close()

print 'End.'

# =============================================================================
