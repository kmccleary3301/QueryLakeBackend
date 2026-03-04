const MARKDOWN_SAMPLE_TEXT = `
# Markdown Example

## Text Formatting

**Bold text**  
*Italic text*  
~~Strikethrough text~~  
[Hyperlink](https://www.example.com)  

## Lists

### Unordered List
- Item 1
- Item 2
  - Subitem 1
  - Subitem 2
    - Subitem 1
    - Subitem 2

### Ordered List
1. First item
2. Second item
   - Subitem A
   - Subitem B


# Kerr Solution

$$
ds^2 = -\\left( 1 - \\frac{2GMr}{\\rho^2 c^2} \\right) c^2 dt^2 - \\frac
{4GMar\\sin^2\\theta}{\\rho^2 c^2} d\\phi dt + \\frac{\\rho^2}{\\Delta} dr^2 + 
\\rho^2 d\\theta^2 + \\left( r^2 + a^2 + \\frac{2GMa^2 r \\sin^2\\theta}{\\rho^2 c^2} \\right) \\sin^2\\theta \\, d\\phi^2,
$$

### Sample Code

\`\`\`python
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample dataset
X = [[1, 2], [2, 3], [3, 4], [4, 5]]  # Features
y = [0, 0, 1, 1]  # Labels

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Gaussian Naive Bayes classifier
clf = GaussianNB()

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
\`\`\`
`

export const MARKDOWN_CHAT_SAMPLE_TEXT = `
## Lorem Ipsum

dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
`

// export MARKDOWN_CHAT_SAMPLE_TEXT;
export default MARKDOWN_SAMPLE_TEXT;