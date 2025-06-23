async function sendQuery() {
    const query = document.getElementById("queryInput").value;

    if (!query.trim()) {
        alert("Please enter a question!");
        return;
    }

    try {
        const response = await fetch("http://127.0.0.1:5000/api/query", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ query: query }),
        });

        if (!response.ok) {
            throw new Error("Failed to fetch response from server");
        }

        const result = await response.json();

        // Hiển thị phản hồi sử dụng Markdown
        const responseDiv = document.getElementById("response");
        responseDiv.innerHTML = marked.parse(result.response || "No response received.");
    } catch (error) {
        console.error("Error:", error);
        document.getElementById("response").textContent = "Something went wrong.";
    }
}
