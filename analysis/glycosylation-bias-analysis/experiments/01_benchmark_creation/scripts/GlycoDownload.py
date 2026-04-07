from SPARQLWrapper import SPARQLWrapper, TURTLE, CSV

endpoint = "https://glyconnect.expasy.org/sparql"

# Example 1: table of proteins and UniProt isoforms as CSV
sparql = SPARQLWrapper(endpoint)
sparql.setQuery("""
PREFIX gc: <https://glyconnect.expasy.org/ontology#>
SELECT DISTINCT ?glycProtein ?uniprotIsoform
WHERE {
  ?glycProtein a gc:Glycoprotein ;
               gc:has_uniprot_isoform ?uniprotIsoform .
}
""")
sparql.setReturnFormat(CSV)
csv_bytes = sparql.query().convert()

with open("protein_ref_isoform.csv", "wb") as f:
  f.write(csv_bytes)

# Example 2: RDF triples (Turtle) in chunks
sparql = SPARQLWrapper(endpoint)
sparql.setReturnFormat(TURTLE)

chunk_size = 10000
offset = 0

with open("glyconnect_all.ttl", "wb") as out:
    while True:
        query = f"""
        CONSTRUCT {{ ?s ?p ?o }}
        WHERE {{ ?s ?p ?o }}
        LIMIT {chunk_size} OFFSET {offset}
        """
        sparql.setQuery(query)
        ttl_bytes = sparql.query().convert()
        if not ttl_bytes.strip():
            break  # no more data
        out.write(ttl_bytes + b"\n")
        offset += chunk_size