#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load((open("../final_project/final_project_dataset.pkl", "rb")), fix_imports=True)

# count = 0
# for v in enron_data.values():
#     if v["poi"]:
#         count += 1
# print(count)
# Jeffrey K Skilling

count_salary = 0
count_email = 0
for p in enron_data:
    if enron_data[p]["salary"] != "NaN":
        count_salary += 1
    if enron_data[p]["email_address"] != "NaN":
        count_email += 1

print("count_salary: ", count_salary)
print("count_email: ", count_email)


