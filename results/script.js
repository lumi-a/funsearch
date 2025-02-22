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

function details(title, maybeContent) {
    const details = document.createElement("details")
    details.classList.add("inner-details")
    const summary = document.createElement("summary")
    summary.textContent = title
    details.appendChild(summary)
    if (maybeContent) {
        details.appendChild(maybeContent)
    }
    return details
}

function detailsCode(title, code) {
    const pre = document.createElement("pre")
    const codeElement = document.createElement("code")
    codeElement.classList.add("language-python")
    codeElement.textContent = code
    pre.appendChild(codeElement)
    hljs.highlightElement(codeElement)
    return details(title, pre)
}

async function displayDatabase(database) {
    /* Schema from generate.py (might be outdated)

     {
        "config": vars(database._config),  # noqa: SLF001
        "inputs": database.inputs,
        "specCode": database._specification,  # noqa: SLF001
        "problemName": database.problem_name,
        "timestamp": database.timestamp,
        "islands": [
          {
            "runs": island._runs,  # noqa: SLF001
            "improvements": [(ix, str(program)) for ix, program in island._improvements],  # noqa: SLF001
            "successCount": island._success_count,  # noqa: SLF001
            "failureCount": island._failure_count,  # noqa: SLF001
          }
          for island in database._islands  # noqa: SLF001
        ],
      },
    
    */
    const problemName = database.problemName
    const problemContainer = getProblemContainer(problemName)

    const runContainer = getRunContainer(problemContainer, problemName, database.inputs, database.timestamp)

    runContainer.appendChild(detailsCode("Spec", database.specCode))

    const islands = database.islands.map((island, i) => {
        const lastImprovement = island.improvements[island.improvements.length - 1]
        island.bestScore = island.runs[lastImprovement[0]]
        island.ix = i
        return island
    })
    islands.sort((a, b) => b.bestScore - a.bestScore)
    const bestProgramsDetails = details("Best Programs", null)
    runContainer.appendChild(bestProgramsDetails)
    islands.forEach(island => {
        const lastCode = island.improvements[island.improvements.length - 1][1]
        bestProgramsDetails.appendChild(detailsCode(`Score ${island.bestScore}, Island ${island.ix}`, lastCode))
    })
    const improvementsDetails = details("Improvements over Time", null)
    runContainer.appendChild(improvementsDetails)
    islands.forEach(island => {
        const islandDetails = details(`Island ${island.ix}`, null)
        improvementsDetails.appendChild(islandDetails)
        for (let imp_ix = island.improvements.length - 1; imp_ix >= 0; imp_ix--) {
            const score = island.runs[imp_ix]
            const run = island.improvements[imp_ix][0]
            const program = island.improvements[imp_ix][1]
            islandDetails.appendChild(detailsCode(`Run ${run}, Score ${score}`, program))
        }
    })


    detailsCode(runContainer, "Config", Object.entries(database.config).map(([k, v]) => `${k} = ${JSON.stringify(v)}`).join("\n"))
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