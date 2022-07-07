from flask import Flask, request, jsonify
import pickle
import numpy as np
import math
from collections import Counter
import json
import pandas as pd

import re
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

## reading the inverted_index file
with open('body_index.pkl', 'rb') as f:
    body_index = pickle.load(f)

## reading the inverted_index file
##with open('title_index.pkl', 'rb') as f:
##    title_index = pickle.load(f)

##with open('anchor_index.pkl', 'rb') as f:
##    anchor_index = pickle.load(f)

##with open('pageviews.pkl', 'rb') as f:
##    page_views = pickle.load(f)
    
import csv
##with open('page_rank.csv', mode='r') as infile:
##    reader = csv.reader(infile)
##    page_rank_dict = {int(rows[0]): float(rows[1]) for rows in reader}

import statistics
##avg_views_values = statistics.mean(list(page_rank_dict.values()))
##avg_rank_values = statistics.mean(list(page_views.values()))

### code from Assignment 4 ###

words_text, pls = zip(*body_index.posting_lists_iter())
print("built and opened all files")

## calculating term_total
term_total_dict = {}
for i in range(len(words_text)):
    sum = 0
    for j in pls[i]:
        sum += j[1]
    term_total_dict[words_text[i]] = sum


###############################################
def generate_query_tfidf_vector(query_to_search, index):
    epsilon = .0000001
    total_vocab_size = len(term_total_dict)
    Q = np.zeros((total_vocab_size))
    term_vector = list(term_total_dict.keys())
    counter = Counter(query_to_search)
    for token in np.unique(query_to_search):
        if token in term_total_dict.keys():  # avoid terms that do not appear in the index.
            tf = counter[token] / len(query_to_search)  # term frequency divded by the length of the query
            df = index.df[token]
            idf = math.log((len(index.DL)) / (df + epsilon), 10)  # smoothing

            try:
                ind = term_vector.index(token)
                Q[ind] = tf * idf
            except:
                pass
    return Q


###############################################
def generate_document_tfidf_matrix(query_to_search, index, words, pls):
    total_vocab_size = len(term_total_dict)
    candidates_scores = get_candidate_documents_and_scores(query_to_search, index, words,
                                                           pls)  # We do not need to utilize all document. Only the docuemnts which have corrspoinding terms with the query.
    unique_candidates = np.unique([doc_id for doc_id, freq in candidates_scores.keys()])
    D = np.zeros((len(unique_candidates), total_vocab_size))
    D = pd.DataFrame(D)

    D.index = unique_candidates
    D.columns = term_total_dict.keys()

    for key in candidates_scores:
        tfidf = candidates_scores[key]
        doc_id, term = key
        D.loc[doc_id][term] = tfidf

    return D


###############################################3
def cosine_similarity(D, Q):
    dictionary = {}
    for index, row in D.iterrows():
        dictionary[index] = (row.dot(Q)) / (math.sqrt(row.dot(row)) * math.sqrt(Q.dot(Q)))
    return dictionary


###############################################3
def get_top_n(sim_dict, N=3):
    return sorted([(doc_id, round(score, 5)) for doc_id, score in sim_dict.items()], key=lambda x: x[1], reverse=True)[
           :N]


###############################################

def get_candidate_documents_and_scores(query_to_search, index, words, pls):
    candidates = {}
    N = len(index.DL)
    for term in np.unique(query_to_search):
        if term in words:
            list_of_doc = pls[words.index(term)]
            normlized_tfidf = [(doc_id, (freq / index.DL[doc_id]) * math.log(N / index.df[term], 10)) for doc_id, freq
                               in list_of_doc]

            for doc_id, tfidf in normlized_tfidf:
                candidates[(doc_id, term)] = candidates.get((doc_id, term), 0) + tfidf

    return candidates


###############################################

def merge_results(title_scores, body_scores, anchor_scores, anchor_weight=0.33, title_weight=0.33, text_weight=0.33,
                  rank_weight=0.5, view_weight=0.5):
    merge_lst = []
    ind_title = 0
    ind_text = 0
    ind_anchor = 0
    title_length = len(title_scores)
    text_length = len(body_scores)
    anchor_length = len(anchor_scores)
    anchor_continue, text_continue, title_continue = (anchor_length != 0), (text_length != 0), (title_length != 0)
    while (anchor_continue + text_continue + title_continue) > 1:  # if only one continue, quit while
        # title stats
        if title_continue:
            # curr_rank_title = (rank_weight * (page_rank_dict[title_scores[ind_title][0]]) / avg_rank_values)
            # curr_views_title = (view_weight * (page_views[title_scores[ind_title][0]]) / avg_views_values)
            curr_title = ((title_length - ind_title) * title_weight)# * (curr_rank_title + curr_views_title)
        # text/body stats
        if text_continue:
            # curr_rank_text = (rank_weight * (page_rank_dict[body_scores[ind_text][0]]) / avg_rank_values)
            # curr_views_text = (view_weight * (page_views[body_scores[ind_text][0]]) / avg_views_values)
            curr_text = ((text_length - ind_text) * text_weight)# * (curr_rank_text + curr_views_text)
        # anchor stats
        if anchor_continue:
            # curr_rank_anchor = (rank_weight * (page_rank_dict[anchor_scores[ind_anchor][0]]) / avg_rank_values)
            # curr_views_anchor = (view_weight * (page_views[anchor_scores[ind_anchor][0]]) / avg_views_values)
            curr_anchor = ((anchor_length - ind_anchor) * anchor_weight)# * (curr_rank_anchor + curr_views_anchor)
        # pick one to add
        if anchor_continue == True and text_continue == True and title_continue == True:
            if curr_title >= curr_text and curr_title >= curr_anchor:
                merge_lst.append(title_scores[ind_title])
                ind_title += 1
            elif curr_text >= curr_anchor:  # curr_text >= curr_title
                merge_lst.append(body_scores[ind_text])
                ind_text += 1
            else:  # curr_anchor bigger that both
                merge_lst.append(anchor_scores[ind_anchor])
                ind_anchor += 1
        elif anchor_continue == True and text_continue == True:
            if curr_text >= curr_anchor:
                merge_lst.append(body_scores[ind_text])
                ind_text += 1
            else:
                merge_lst.append(anchor_scores[ind_anchor])
                ind_anchor += 1
        elif text_continue == True and title_continue == True:
            if curr_title >= curr_text:
                merge_lst.append(title_scores[ind_title])
                ind_title += 1
            else:
                merge_lst.append(body_scores[ind_text])
                ind_text += 1
        else:  # anchor_continue == True and title_continue == True
            if curr_title >= curr_anchor:
                merge_lst.append(title_scores[ind_title])
                ind_title += 1
            else:
                merge_lst.append(anchor_scores[ind_anchor])
                ind_anchor += 1

        anchor_continue, text_continue, title_continue = (ind_anchor != anchor_length), (ind_text != text_length), \
                                                         (ind_title != title_length)
    merge_lst += title_scores[ind_title:]
    merge_lst += body_scores[ind_text:]
    merge_lst += anchor_scores[ind_anchor:]
    return merge_lst[:100]


###############################################



search_query = None
search_flag = False


def set_query(new_query):
    global search_query
    search_query = new_query


@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    # res = []
    # query = request.args.get('query', '')
    # print(query)
    # if len(query) == 0:
    #     return jsonify(res)
    # # BEGIN SOLUTION

    # # END SOLUTION
    # return jsonify(res)



    ### barak solution
    res = []
    query = request.args.get('query', '')
    global search_flag
    search_flag = True
    global search_query
    # query = search_query
    set_query(query)
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    titles = search_title()
    body = search_body()
    anchor = search_anchor()
    res = merge_results(titles, body, anchor, anchor_weight=0.33, title_weight=0.33, text_weight=0.33, rank_weight=0.5,
                        view_weight=0.5)
    # END SOLUTION
    ids = []
    for tup in res:
        ids.append(tup[0])

    search_flag = False
    # return res
    return jsonify(res[:100])


@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. 

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    # query = request.args.get('query', '')

    global search_flag
    if search_flag:
      global search_query
      query = search_query
    else:
      query = request.args.get('query', '')
    
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    query = query.lower().split()
    # print ("data: ",data.df)
    query_idf_vector = generate_query_tfidf_vector(query, body_index)
    doc_idf = generate_document_tfidf_matrix(query, body_index, words_text, pls)
    calculate_cosSim = cosine_similarity(doc_idf, query_idf_vector)
    sorted_by_score = get_top_n(calculate_cosSim, 100)
    for t in sorted_by_score:  # i is tuple (wiki_id,score)
        tup = (t[0], title_index.id_to_title[t[0]])  # need to get the title by the wiki_id
        res.append(tup)
    # END SOLUTION
    if search_flag:
        return res
    return jsonify(res)


@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        QUERY WORDS that appear in the title. For example, a document with a 
        title that matches two of the query words will be ranked before a 
        document with a title that matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    # query = request.args.get('query', '')

    global search_flag
    if search_flag:
      global search_query
      query = search_query
    else:
      query = request.args.get('query', '')

    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    titles = []
    # tokens = [token.group() for token in RE_WORD.finditer(query.lower())]
    # query = tokens
    query = query.lower().split()
    for word in query:
      if word in title_index.word_to_titles:
        titles += title_index.word_to_titles[word]
    for title in titles:
      if title not in res:
        res.append(title)
    # END SOLUTION
    if search_flag:
      return res
    return jsonify(res)


def number_of_anchors(docID_anchorID, identifier):
    count = 0
    for tup in docID_anchorID:
        if identifier == tup[0]:
            count += 1
    return count


@app.route("/search_anchor")
def search_anchor():
    """ Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE ANCHOR TEXT of articles, ordered in descending order of the
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page.
        For example, a document with a anchor text that matches two of the
        query words will be ranked before a document with anchor text that
        matches only one query term.

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    """
    res = []
    # query = request.args.get('query', '')

    global search_flag
    if search_flag:
      global search_query
      query = search_query
    else:
      query = request.args.get('query', '')

    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    docID_anchorID = []
    # data = anchor_index.term_to_ids["data"]
    # science = anchor_index.term_to_ids["science"]
    tokens = [token.group() for token in RE_WORD.finditer(query.lower())]
    for word in tokens:
        if word in anchor_index.term_to_ids:
            docID_anchorID += anchor_index.term_to_ids[word]

    anchorID = []
    for tup in docID_anchorID:
        if tup[0] not in anchorID:
            anchorID.append(tup[0])
    res = sorted(anchorID, key=lambda x: number_of_anchors(docID_anchorID, x), reverse=True)
    for index in range(len(res)):
        res[index] = (res[index], title_index.id_to_title[res[index]])
    # END SOLUTION
    if search_flag:
      return res
    return jsonify(res)


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    # for identifier in wiki_ids:
    #     res.append(page_rank_dict[identifier])

    # END SOLUTION
    return jsonify(res)


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    for id in wiki_ids:
        res.append(page_views[id])
    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
