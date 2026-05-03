def infer_cluster_name(cluster_id):
    cluster_names = {
        0: "Cloud Computing and Web Services",
        1: "General CS, Formal Methods and Systems",
        2: "Computer Vision and Neural Learning",
        3: "Data Mining, Databases and Knowledge Discovery",
        4: "Optimization and Evolutionary Computation",
        5: "Information Systems and Human-Centered Computing",
        6: "Bioinformatics and Computational Biology",
        7: "Communication Networks and Telecommunications",
        8: "Fuzzy Logic and Decision Making",
        9: "Wireless Sensor Networks and Mobile Systems",
    }
    return cluster_names.get(cluster_id, f"Cluster {cluster_id}")


def get_cluster_keywords(cluster_id):
    cluster_terms = {
        0: ["cloud", "service", "computing", "web", "iot"],
        1: ["software", "systems", "theory", "methods", "model"],
        2: ["image", "recognition", "neural", "learning", "detection"],
        3: ["data", "mining", "clustering", "query", "big data"],
        4: ["optimization", "algorithm", "genetic", "evolutionary", "swarm"],
        5: ["information", "social", "user", "business", "knowledge"],
        6: ["biology", "gene", "protein", "molecular", "bioinformatics"],
        7: ["networks", "wireless", "telecommunications", "routing", "traffic"],
        8: ["fuzzy", "decision", "rough sets", "uncertainty", "intuitionistic"],
        9: ["sensor", "wireless sensor", "nodes", "network", "wsn"],
    }
    return cluster_terms.get(cluster_id, [])
