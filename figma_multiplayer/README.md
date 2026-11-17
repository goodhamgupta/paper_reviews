# Figma multiplayer

- Not built on operational transforms(OTs)
  - Too complex for their problem space
- Custom system that's 'simplier'

## Background

- Client-server architecture
- Figma clients are web pages that talk to servers over websockets.
- [Scale web services using rust](https://www.figma.com/blog/rust-in-production-at-figma/)

- When document is opened:
  - Client downloads a copy of the file
  - Updates in both directions are synced over ws.
  - When you go offline, and come back, client downloads a new copy of the file, reapplies your updates, and then continue syncing updates over ws.
- Multiplayer system is only used to sync changes to the design files
  - other sync, such as comments, users, etc. is synced via postgres, not multiplayer system.
  - separate client and system for this.
  - different systems because of tradeoffs in perf, offline availability and security. 

## OTs and CRDTs

- OTs
  - overkill for what we needed.
  - good for editing long docs with low memory and overhead, BUT complex and hard to implement.
  - because order of events + actors are different, difficult to predict all states. formal proofs are tricky to reason about.
- Figma uses CRDTs
  - conflict-free replicated datatypes
  - commmonly used in dist systems
  - CRDTs guarantee eventual consistency.
    - important for figma: need design file to eventually show the same content to all clients.
- Types of CRDTs:
  - Grow-only set: 
    - set of elements
    - only update => add something to set
    - adding twice is a no-op
    - can determine contents of the set by just applying in-order. Sort of like a WAL? 
  - Last-writer-wins register:
    - container for a single value.
    updates can be implemented as a new value, timestamp or peer ID.
    - determine value by just looking at the latest update(use peer ID to break a tie.)

- Figma doesn't use 'pure' CRDTs though.
  - 'Pure' CRDTs are needed for decentralized workloads i.e no central authority to determine the final state.
  - unavoidable perf and mem overhead with the above approach.
  - Figma is centralised i.e figma server is the authority, so we can simplify our system by removing the extra overhead. faster and leaner implementation.

- Figma's data structure is NOT A SINLE CRDT. IT's multiple separate CRDTs that are used in combo to create a final data structure called **Figma Document**

## Figma Document

- Every figma document is a tree of objects, similar to HTML DOM.
  - single root that represents entire doc.
  - under root are page objections
  - under page are hierachy of other objects representing content
- Each object has an ID + props
  - Document is a two-level map: <Map<ObjectID, Map<Property, Value>>
  - can also be stored in a db as (ObjectID, Property, Value) tuples.

## Figma Multiple System

### Syncing object props

- Multiplayer servers keep track of latest value any client has sent for a given property on a objeect
- two clients changing unrelated props on same object will not conflict, AND changing same prop will also not conflict(sicne we only keep latest)
  - Basically last-writer-wins approach
  - don't need timestamp because server can define the order of events i.e latest one is the last request received.
- Because of the above, changes are a property level are atomic
  - changing the text field simultaneously by 2 clients DOESNT WORK because of this hah.

- Complicated: how to handle conflicts?
  - when there's a conflicting change, applied immediately instead of waiting for ack from server.
  - BUT, if we get older changes from server and then apply to local, might see 'flciker' i.e old values suddenly visible locally.
  - WANT TO AVOID FLICKERING BEHAVIOUR.
- TLDR: if the local changes are newer, and the changes from the server are conflicting from the local changes, just choose to discard the server changes, since locally, last-writer-wins will apply?(This is my understanding so far)

### Syncing object creation and removal

- modelled as separate explicit actions in the prorocal
- object creation is like last-writer-wins
  - BUT, figma doesn't store props of deleted objects on server
  - Data is stored un the undo bufffer of the client, that performed the delete
  - if they wanna undo, then client is repsonsible for restoring all properties of the deleted objects.
- system relies on clients being able to generate unique object IDs that are guaranteed to be unique
  - assign every client unique client ID, and include client ID as part of the object ID.
  - object ID creation will be on the client, NOT THE SERVER SIDE, since object ids creation needs to work offline as well.

### Syncing trees of objects i.e reparenting

- Reparenting object should not conflict with changes unrelated to props on those objects
  - if someone change object color, while someone else reparents, both ops should succeed.
- two concurrent reparenting ops for same object should never result in two copes of that object in different places in the tree.

- tradional approach: represent reparenting as delete + add
  - doesn't work for figma because concurrent edits would be dropped when object ID changes.
- Solution: Represent parent-child relationship by storing link to the parent as a property on the child.
  - this way, object identity is preseved.
  - also don't need to deal with case where object somehow ends up with multiple parents.

- HOWEVER, NOW NEW PROBLEM: without any other restrictions, these parent links are just directed edges on a graph.
  - HOW TO ENSURE THERE ARE NO CYCLES?
  - Figma server rejects props that cause a cycle on the server side, but cycles CAN STILL HAPPEN ON CLIENT SIDE!
  - Client cannot reject changes since there is no 'ultimate' authority that knows the final state.
- Figma's solution: 
  - temporarily parent these objects to each other and remove them from the tree untill server rejects the client's change and object is reparented where it belongs.

- To construct the tree:
  - need to detemrine order of children.
  - Use **fractional indexing**: objects position in its parent's array of children is represented as a fraction between 0 and 1 exclusive
  - order of object's children is determined by sorting them by their positions.
  - can insert object between two objects by setting it's position to average of the positions of the two objects.

- **NOTE**: Parent link + this position must be stored as a single property WITH ATOMIC UPDATES ONLY.

## Fractional Indexing

Review of article: https://www.figma.com/blog/realtime-editing-of-ordered-sequences/#fractional-indexing

- Implement reordering on top of a db.
- every obj has a real number as a index(i.e num between 0 to 1 exclusive)
  - exclusive is important since we want to be able to generate an index if an object is moved to the start or end of the graph.
- order determined by storting all children by index
- insert: take average of two objects and insert in between.
- use 64-bit doubles so we can't run out of precision.

- Pros:
  - easy to understand and implement
  - reodering an object involves editing a single value

- Cons:
  - index len can grow over time
    - not a concern in practice, since reordering ops depend on user behaviour, and usually don't see crazy-high reorderings.
  - merging new elements from multiple clients may interleave them
    - concern for IDE but not for figma given nature of design docs.
    - interleaving concurrently designed elements is fine because new objects usually dont overlap.
    - if it looks weird, users can fix it manually.
  - averaging two identical indices doesn't work
    - happens when two clients try to simultaneously insert a new object between the same two objects.
    - solution: server can avoid by generating and assigning a unique position to the second insert op, as part of reconcilliation
