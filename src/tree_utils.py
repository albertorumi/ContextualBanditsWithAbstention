class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        self.leaf = False
        self.graphic_rappr = None

def insert_perfect_binary_tree(root, values, index):
    if index < len(values):
        root = TreeNode(values[index])
        root.left = insert_perfect_binary_tree(root.left, values, 2 * index + 1)
        root.right = insert_perfect_binary_tree(root.right, values, 2 * index + 2)
    return root

def create_perfect_binary_tree(values):
    if not values:
        return None
    return insert_perfect_binary_tree(None, values, 0)

def find_parents(root, target_value):
    def dfs_fp(node, parent, result):
        if not node:
            return

        if node.value == target_value:
            result.add(parent.value if parent else None)
            return

        for child in [node.left, node.right]:
            dfs_fp(child, node, result)

    result = set()
    dfs_fp(root, None, result)
    return result


def nodes_at_depth(root, target_depth, current_depth=0):
    if root is None:
        return []

    if current_depth == target_depth:
        return [root]

    left_nodes = nodes_at_depth(root.left, target_depth, current_depth + 1)
    right_nodes = nodes_at_depth(root.right, target_depth, current_depth + 1)

    return left_nodes + right_nodes

def find_parents(root, target):
    def dfs(node, parent, path):
        if not node:
            return None

        if node.value == target:
            return path

        left_path = dfs(node.left, node, path + [node])
        right_path = dfs(node.right, node, path + [node])

        return left_path if left_path else right_path

    return dfs(root, None, [])#

def print_tree(root, level=0, prefix="Root: "):
    if root is not None:
        print(" " * (level * 4) + prefix + str(root.value))
        if root.left or root.right:
            print_tree(root.left, level + 1, "L--- ")
            print_tree(root.right, level + 1, "R--- ")

def is_perfect_tree(root):
    if not root:
        return True
    
    # Get the height of the leftmost path
    height = 0
    current = root
    while current:
        height += 1
        current = current.left
    
    # Check if the tree is perfect
    def is_perfect_recursive(node, current_level):
        if not node:
            return True
        # If a node has only one child or no child, it's not perfect
        if (node.left is None and node.right) or (node.left and node.right is None):
            return False
        # If we reach a leaf node and it's at the expected level, it's perfect
        if current_level == height:
            return True
        # Recursively check left and right subtrees
        return (is_perfect_recursive(node.left, current_level + 1) and
                is_perfect_recursive(node.right, current_level + 1))

    return is_perfect_recursive(root, 1)