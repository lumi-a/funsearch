const jsonDir = "json-data"

async function loadAllData() {
    const indexResponse = await fetch(jsonDir + "/index.json")
    const files = await indexResponse.json()

    const datasets = await Promise.all(files.map(async (file) => {
        const response = await fetch(jsonDir + "/" + file)
        return response.json()
    }))

    return datasets
}

function strToPre(str) {
    pre = document.createElement("pre")
    code = document.createElement("code")
    code.classList.add("language-python")
    code.textContent = database[0]["fn"]
    pre.appendChild(code)
    hljs.highlightElement(code)
    return pre
}

function getProblemContainer(problemName) {
    const maybeExisting = document.getElementById("container-" + problemName)
    if (maybeExisting) {
        return maybeExisting
    }

    const div = document.createElement("div")
    div.id = "container-" + problemName
    document.body.appendChild(div)

    const heading = document.createElement("h2")
    heading.textContent = problemName
    div.appendChild(heading)

    return div
}

function getRunContainer(problemContainer, problemName, inputs, timestamp) {
    const details = document.createElement("details")
    // Let's just hope these are unique.
    details.id = "run-" + timestamp

    const summary = document.createElement("summary")

    const problemSpan = document.createElement("span")
    problemSpan.textContent = problemName + "(" + inputs.join(", ") + ")"
    summary.appendChild(problemSpan)

    const timestampLink = document.createElement("a")
    timestampLink.classList.add("timestamp")
    timestampLink.href = `#run-${timestamp}`
    timestampLink.textContent = timestamp
    summary.appendChild(timestampLink)

    details.appendChild(summary)

    problemContainer.appendChild(details)

    return details
}

async function displayDatabase(database) {
    /*{
        "config": vars(database._config),  # noqa: SLF001
          "inputs": database.inputs,
        "specCode": database._specification,  # noqa: SLF001
        "failureCounts": database._failure_counts,  # noqa: SLF001
        "successCounts": database._success_counts,  # noqa: SLF001
        "bestScorePerIsland": database._best_score_per_island,  # noqa: SLF001
        "bestProgramPerIsland": [str(p) for p in database._best_program_per_island],  # noqa: SLF001
          "problemName": database.problem_name,
          "timestamp": database.timestamp,
    },*/
    const problemName = database.problemName
    const problemContainer = getProblemContainer(problemName)

    const runContainer = getRunContainer(problemContainer, problemName, database.inputs, database.timestamp)

}

async function main() {
    const databases = await loadAllData()
    databases.forEach(displayDatabase)
}

hljs.addPlugin(new CopyButtonPlugin())

window.addEventListener('load', () => {
    main().then(() => {
        for (let i = 0; i < 100; i++) {
            document.body.appendChild(document.createTextNode("br"))
            document.body.appendChild(document.createElement("br"))
        }

        const hash = window.location.hash.slice(1)
        if (hash) {
            console.log(hash)
            const element = document.getElementById(hash)
            if (element) {
                element.open = true
                element.scrollIntoView()
            }
        }
    })
})