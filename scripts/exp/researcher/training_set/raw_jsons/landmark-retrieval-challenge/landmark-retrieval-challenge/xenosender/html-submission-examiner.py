# This script generates a HTML file that represents the submission file you're sending
# it shows, for the 50 first doc in test, the 30 first prediction images so that you can control 
# easily if they match the test document
# All images are clickable

# The css to use is :
#    .bordered {
#        border: 1px solid black
#    }
#    
#    .predictions_div {
#        'overflow-x': 'scroll';
#        width="700px";
#    }
#    
#    .predictions_table {
#        border-collapse: collapse;
#        display: block;
#        width=2000px;
#    }
#    
#    ul {
#        list-style-type: none;
#    }
#
# To display the html, go in the report directory and run "python -m SimpleHTTPServer 8000"

from yattag import Doc, indent
import csv
import os

doc, tag, text = Doc().tagtext()

doc.asis('<!DOCTYPE html>')


def parse_data(data_file):
    """
    read index files 
    """
    csvfile = open(data_file, 'r')
    csvreader = csv.reader(csvfile)
    key_url_list = {line[0]: line[1] for line in csvreader}
    return key_url_list


def generate_image_link(key, key_url_list):
    """
    generate element for 1 image
    """
    url = key_url_list[key]
    with tag('div', width='120px'):
        with tag('ul'):
            with tag('li'):
                with tag('a', href=url, target="_blank"):
                    doc.stag('img', src=url, klass="photo", width='100')
            with tag('li'):
                text(key)


def generate_doc_images_list(doc_id, prediction_lists, key_url_list):
    """
    generate 1 line (example + predictions)
    """
    with tag('table'):
        with tag('tr'):
            with tag("td", klass="bordered"):
                generate_image_link(doc_id, key_url_list)
            with tag('td', klass="bordered"):
                with tag('div', klass="predictions_div"):
                    with tag('table', klass="predictions_table"):
                        with tag('tr'):
                            for pred in prediction_lists:
                                with tag('td'):
                                    generate_image_link(pred, key_url_list)



if __name__ == "__main__":

    # input files
    data_index_file = "../input/index.csv"
    key_url_list = parse_data(data_index_file)

    data_test_file = "../input/test.csv"
    key_url_list_test = parse_data(data_test_file)
    key_url_list.update(key_url_list_test)

    # Predictions
    report_file = "../input/sample_submission.csv"
    docs = []
    prediction_lists = {}
    with open(report_file) as f:
        start = True
        for line in f.readlines():
            if start:
                start = False
                continue
            test_id, predictions = line.split(",")
            predictions = predictions.split()
            docs.append(test_id)
            prediction_lists[test_id] = predictions

    # generate html
    with tag('html'):
        with tag('head'):
            doc.asis('<meta charset="utf-8">')
            doc.asis('<meta name="viewport" content="width=device-width, initial-scale=1">')
            doc.asis('<link rel="stylesheet" href="http://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/css/bootstrap.min.css">')
            doc.asis('<link rel="stylesheet" href="report.css">')
            with tag('script', src="https://ajax.googleapis.com/ajax/libs/jquery/1.12.0/jquery.min.js"):
                pass
            with tag('script', src="http://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/js/bootstrap.min.js"):
                pass
        with tag('body'):
            with tag("h1"):
                text("Report for file {}".format(os.path.basename(report_file)))

            for doc_id in docs[:50]:
                generate_doc_images_list(doc_id, prediction_lists[doc_id][:30], key_url_list)

    # save html
    output_file = os.path.basename(report_file[:report_file.rfind('.')] + ".html")
    with open(output_file, 'w') as f:
        f.write(indent(doc.getvalue()))
        
    
    css_text = """
    .bordered {
        border: 1px solid black;
    }    
    .predictions_div {
        'overflow-x': 'scroll';
        width="700px";
    }
    .predictions_table {
        border-collapse: collapse;
        display: block;
        width=2000px;
    }
    ul {
        list-style-type: none;
    }"""
    with open("report.css", 'w') as f:
        f.write(css_text)