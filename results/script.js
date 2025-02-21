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
    div.classList.add("problem-container")
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
    details.classList.add("run-container")

    const summary = document.createElement("summary")

    const problemSpan = document.createElement("code")
    problemSpan.textContent = problemName + "(" + inputs.join(", ") + ")"
    summary.appendChild(problemSpan)

    const timestampLink = document.createElement("a")
    timestampLink.classList.add("timestamp")
    timestampLink.href = `#run-${timestamp}`
    timestampLink.textContent = timestamp
    summary.appendChild(timestampLink)

    details.appendChild(summary)

    const containerInner = document.createElement("div")
    containerInner.classList.add("run-container-inner")
    details.appendChild(containerInner)

    problemContainer.appendChild(details)

    return containerInner
}

function appendDetails(container, title, content) {
    const details = document.createElement("details")
    details.classList.add("inner-details")
    const summary = document.createElement("summary")
    summary.textContent = title
    details.appendChild(summary)
    details.appendChild(content)
    container.appendChild(details)
}

function appendDetailsCode(container, title, code) {
    const pre = document.createElement("pre")
    const codeElement = document.createElement("code")
    codeElement.classList.add("language-python")
    codeElement.textContent = code
    pre.appendChild(codeElement)
    hljs.highlightElement(codeElement)
    appendDetails(container, title, pre)
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

    appendDetailsCode(runContainer, "Spec", database.specCode)

    appendDetailsCode(runContainer, "Config", Object.entries(database.config).map(([k, v]) => `${k} = ${JSON.stringify(v)}`).join("\n"))
}

async function main() {
    const databases = await loadAllData()
    databases.forEach(displayDatabase)
}

hljs.addPlugin(new CopyButtonPlugin())

window.addEventListener('load', () => {
    main().then(() => {
        const hash = window.location.hash.slice(1)
        if (hash) {
            const element = document.getElementById(hash)
            if (element) {
                element.open = true
                element.scrollIntoView()
            }
        }
    })
})