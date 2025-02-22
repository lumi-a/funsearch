const jsonDir = "json-data"

async function loadAllData() {
    const indexResponse = await fetch(jsonDir + "/index.json")
    const files = await indexResponse.json()

    const datasets = await Promise.all(files.map(async (file) => {
        const response = await fetch(`${jsonDir}/${file}`)
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

function improvementCanvas(islands, highestRunIndex) {
    const improvementsCanvas = document.createElement("canvas")
    new Chart(
        improvementsCanvas,
        {
            type: 'scatter',
            data: {
                datasets: islands.map(island => {
                    // Manually create step chart
                    let x = island.improvements[0][0]
                    let y = island.improvements[0][1]
                    let data = [{ x, y }]
                    for (let i = 1; i < island.improvements.length; i++) {
                        x = island.improvements[i][0]
                        data.push({ x, y }) // [sic], creates step-chart
                        y = island.improvements[i][1]
                        data.push({ x, y })
                    }
                    data.push({ x: highestRunIndex, y })
                    return {
                        label: `Island ${island.ix}`,
                        data: data,
                        showLine: true,
                    }
                }
                ),
            },
            options: {
                scales: {
                    x: {
                        display: true,
                        title: {
                            display: true,
                            text: "Run Index"
                        }
                    },
                    y: {
                        display: true,
                        title: {
                            display: true,
                            text: "Running Max Score"
                        }
                    }
                }
            }
        }
    )
    return improvementsCanvas
}

function getProblemContainer(problemName) {
    const maybeExisting = document.getElementById(`container-${problemName}`)
    if (maybeExisting) {
        return maybeExisting
    }

    const div = document.createElement("div")
    div.id = `container-${problemName}`
    div.classList.add("problem-container")
    document.body.appendChild(div)

    const heading = document.createElement("h2")
    heading.textContent = problemName
    div.appendChild(heading)

    return div
}

function details(title, description, ...content) {
    const details = document.createElement("details")
    const summary = document.createElement("summary")
    details.appendChild(summary)

    const containerInner = document.createElement("div")
    containerInner.classList.add("details-inner")
    details.appendChild(containerInner)

    const titleSpan = document.createElement("span")
    titleSpan.textContent = title
    titleSpan.classList.add("title")
    summary.appendChild(titleSpan)

    if (description) {
        const descriptionSpan = document.createElement("span")
        descriptionSpan.textContent = description
        descriptionSpan.classList.add("description")
        summary.appendChild(descriptionSpan)
    }

    content.forEach(elem => containerInner.appendChild(elem))
    return details
}

function detailsCode(title, description, code) {
    const pre = document.createElement("pre")
    const codeElement = document.createElement("code")
    codeElement.classList.add("language-python")
    codeElement.textContent = code
    pre.appendChild(codeElement)
    hljs.highlightElement(codeElement)
    return details(title, description, pre)
}

// Paul Tol's discrete rainbow color scheme, from https://personal.sron.nl/~pault/
const colors = ['#CC6677', '#332288', '#DDCC77', '#117733', '#88CCEE', '#882255', '#44AA99', '#999933', '#AA4499', '#888']

async function displayDatabase(database) {
    /* Schema from generate.py (might be outdated)
      {
        "problemName": database.problem_name,
        "inputs": database.inputs,
        "message": database.message,
        "config": vars(database._config),  # noqa: SLF001
        "specCode": database._specification,  # noqa: SLF001
        "timestamp": database.timestamp,
        "highestRunIndex": max(len(island._runs.keys()) for island in database._islands),  # noqa: SLF001
        "islands": [
          {
            "improvements": [(ix, island._runs[ix], str(program)) for ix, program in island._improvements],  # noqa: SLF001
            "successCount": island._success_count,  # noqa: SLF001
            "failureCount": island._failure_count,  # noqa: SLF001
          }
          for island in database._islands  # noqa: SLF001
        ],
      },
    */
    const problemName = database.problemName
    const problemContainer = getProblemContainer(problemName)

    let totalSuccesses = 0
    let totalFailures = 0
    const islands = database.islands.map((island, i) => {
        const lastImprovement = island.improvements[island.improvements.length - 1]
        island.bestScore = lastImprovement[1]
        island.ix = i
        totalSuccesses += island.successCount
        totalFailures += island.failureCount
        island.rate = (island.successCount + island.failureCount) > 0 ? Math.round(100 * island.failureCount / (island.successCount + island.failureCount)) : 0

        return island
    })
    const totalRate = (totalSuccesses + totalFailures) > 0 ? Math.round(100 * totalFailures / (totalSuccesses + totalFailures)) : 0
    islands.sort((a, b) => b.bestScore - a.bestScore)
    const maxScore = islands[0].bestScore


    const runDetails = details(`${problemName}(${database.inputs.join(', ')}) → ${maxScore}`, database.message,
        detailsCode("Spec", "Specification-file and seed-function", database.specCode),
        details("Best Programs", "Best program of each island", ...islands.map(island => detailsCode(`Score ${island.bestScore}`, `Island ${island.ix}`, island.improvements[island.improvements.length - 1][2])
        )),
        details("Improvements over Time", "Improvement-steps of each island", document.createTextNode(`Total failure-rate: ${totalRate}%`), improvementCanvas(islands, database.highestRunIndex),
            ...islands.map(island => details(`Score ${island.bestScore}`, `Island ${island.ix}, failure-rate ${island.rate}%`,
                ...island.improvements.toReversed().map(improvement => detailsCode(`Score ${improvement[1]}`, `Run ${improvement[0]}`, improvement[2])))
            )),
        detailsCode("Config", "Config-file for this run", Object.entries(database.config).map(([k, v]) => `${k} = ${JSON.stringify(v)}`).join("\n")),
    )
    problemContainer.appendChild(runDetails)

    // Let's just hope these are unique.
    runDetails.id = `run-${database.timestamp}`
    runDetails.classList.add("run-container")

    const messageSpan = document.createElement("span")
    messageSpan.textContent = database.message
    messageSpan.classList.add("pre-wrap")
    const runDetailsInner = runDetails.querySelector(".details-inner")
    runDetailsInner.insertBefore(messageSpan.cloneNode(true), runDetailsInner.firstChild)

    const timestampLink = document.createElement("a")
    timestampLink.classList.add("timestamp")
    const href = `#run-${problemName}-${database.timestamp}`
    timestampLink.href = href
    timestampLink.textContent = href
    runDetails.querySelector("summary").appendChild(timestampLink)
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