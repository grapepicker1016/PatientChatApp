from neo4j import GraphDatabase

# Connect to the Neo4j database


class Neo4jCRUD:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    # CREATE operation
    def create_node(self, name, age):
        with self.driver.session() as session:
            session.write_transaction(self._create_and_return_node, name, age)

    @staticmethod
    def _create_and_return_node(tx, name, age):
        query = (
            "CREATE (p:Person {name: $name, age: $age}) "
            "RETURN p"
        )
        result = tx.run(query, name=name, age=age)
        return result.single()[0]

    # READ operation
    def read_all_nodes(self):
        with self.driver.session() as session:
            result = session.read_transaction(self._get_all_nodes)
            return result

    @staticmethod
    def _get_all_nodes(tx):
        query = "MATCH (p:Person) RETURN p.name AS name, p.age AS age"
        result = tx.run(query)
        return [{"name": record["name"], "age": record["age"]} for record in result]

    # UPDATE operation
    def update_node(self, old_name, new_name):
        with self.driver.session() as session:
            session.write_transaction(
                self._update_node_name, old_name, new_name)

    @staticmethod
    def _update_node_name(tx, old_name, new_name):
        query = (
            "MATCH (p:Person {name: $old_name}) "
            "SET p.name = $new_name "
            "RETURN p"
        )
        tx.run(query, old_name=old_name, new_name=new_name)

    # DELETE operation
    def delete_node(self, name):
        with self.driver.session() as session:
            session.write_transaction(self._delete_node_by_name, name)

    @staticmethod
    def _delete_node_by_name(tx, name):
        query = (
            "MATCH (p:Person {name: $name}) "
            "DELETE p"
        )
        tx.run(query, name=name)


if __name__ == "__main__":
    # Connect to your local Neo4j database (default: bolt://localhost:7687)
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "P@ssw0rd"  # Replace with your password

    neo4j_crud = Neo4jCRUD(uri, user, password)

    # CREATE a node
    neo4j_crud.create_node(name="Alice", age=30)
    neo4j_crud.create_node(name="Bob", age=25)

    # READ all nodes
    print("All Persons:")
    nodes = neo4j_crud.read_all_nodes()
    for node in nodes:
        print(node)

    # UPDATE a node's name
    neo4j_crud.update_node(old_name="Alice", new_name="Alicia")

    # READ all nodes after the update
    print("\nPersons after update:")
    nodes = neo4j_crud.read_all_nodes()
    for node in nodes:
        print(node)

    # DELETE a node
    neo4j_crud.delete_node(name="Alicia")

    # READ all nodes after deletion
    print("\nPersons after deletion:")
    nodes = neo4j_crud.read_all_nodes()
    for node in nodes:
        print(node)

    # Close the connection
    neo4j_crud.close()
