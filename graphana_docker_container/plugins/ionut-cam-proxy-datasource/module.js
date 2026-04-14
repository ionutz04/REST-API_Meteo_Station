"use strict";
System.register(["@grafana/data"], function (exports_1, context_1) {
    "use strict";
    var data_1, plugin;
    var __moduleName = context_1 && context_1.id;
    return {
        setters: [
            function (data_1_1) {
                data_1 = data_1_1;
            }
        ],
        execute: function () {
            exports_1("plugin", plugin = new data_1.AppPlugin());
        }
    };
});
