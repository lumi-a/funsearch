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

function problemNameToContainer(problemName) {
    const maybeExisting = document.getElementById(problemName + "-container")
    if (maybeExisting) {
        return maybeExisting
    }

    const details = document.createElement("details")
    details.id = problemName + "-container"
    document.body.appendChild(details)

    const summary = document.createElement("summary")
    summary.textContent = problemName
    details.appendChild(summary)

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
    const container = problemNameToContainer(problemName)

    // document.body.appendChild(pre)
}

async function main() {
    const databases = await loadAllData()
    databases.forEach(displayDatabase)
}

hljs.addPlugin(new CopyButtonPlugin())
main()