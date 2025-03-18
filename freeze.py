from funny import app
from flask_frozen import Freezer

freezer = Freezer(app)

@freezer.register
def index():
    return app.view_functions['index']()

@freezer.register
def home_kmeans():
    return app.view_functions['home_kmeans']()

@freezer.register
def query_cluster_kmeans():
    return app.view_functions['query_cluster_kmeans']()

@freezer.register
def get_random_document_kmeans():
    return app.view_functions['get_random_document_kmeans']()

@freezer.register
def get_cluster_docs_kmeans(k):
    return app.view_functions['get_cluster_docs_kmeans'](k)

if __name__ == '__main__':
    freezer.freeze()