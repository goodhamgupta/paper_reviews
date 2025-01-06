# Redis Vector Sets

Video: https://www.youtube.com/watch?v=t1JjV4OTgGw&ab_channel=SalvatoreSanfilippo
- Built on HNSW (Hierarchical Navigable Small World) algorithm
- Graph with multiple levels of hierarchy. Each level has few nodes. The closest nodes are connected in the next level.
- Sparse connection graph

Problem:
- Insertion time of HNSW is not great
- EXTREMELY SLOW INSERTION.
- Need to scan the graph with a greedy algorithm to identify the closest nodes.
- HNSW is serialised before storing in Redis.
- After loading embeddings, you can find similar embeddings using the VSIM command in redis.
- You can use more effort to get a better result i.e wait for longer for a more accurate result. In redis, this is done using the EXPLORE command.


## Implementation

- Graph is serialised
- Nodes are in a linked list.
- Node has a  vector.
- Serialize the node as a set of numbers i.e 64-bit number
- For each level, store the number of links, max_links, and the linked node's ID
- The links pointer only stores the ID's of the nodes, not the nodes itself.
- Because the links are not actual nodes, the retrieval will never work.
- HENCE, after loading the serialised nodes, we need to resolve the ID's to the actual nodes.
- Create a hashtable, store the nodes via ID's in the first pass, and in the 2nd pass, fix all links.
- Node ID hash is defined using murmurhash3 algorithm.

