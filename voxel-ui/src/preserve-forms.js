"use strict";

var $ = require('jquery');

function saveFormData() {
    var data = {};
    $('.save').each(function(){
        var $el = $(this);
        var value;

        if ($el.attr('id') === undefined) {
            console.warn('Element cannot be saved because it has no id:', $el)
            return;
        }

        if ($el.is(':checkbox') || $el.is(':radio')) {
            value = $el.prop('checked');
        } else {
            value = $el.val();
        }
        data[$el.attr('id')] = value;
    });

    console.log('saving form data', data);
    localStorage.setItem('formData', JSON.stringify(data));
}

function restoreFormData() {
    var data = JSON.parse(localStorage.getItem('formData'));
    if (data === null) {
        return;
    }

    console.log('restoring form data', data);

    var $el;
    for (var elementId in data) {
        if (data.hasOwnProperty(elementId)) {
            $el = $('#' + elementId);
            if ($el.is(':checkbox') || $el.is(':radio')) {
                $el.prop('checked', data[elementId]);
            } else {
                $el.val(data[elementId]);
            }
        }
    }
}

module.exports = function() {
    restoreFormData();

    $('.save').click(function(){
        saveFormData();
    })
}
